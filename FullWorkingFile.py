#!/usr/bin/env python3
"""
PI PIPELINE + MULTI-SENSOR OBSTACLE DETECTION  (runs on the Raspberry Pi)

OBJECT DETECTION (event-driven, main thread)
    Three buttons, one per object. Press -> camera frame -> YOLO checks ONLY
    for that object -> result printed AND spoken via the DFPlayer (object clip,
    then "detected, N o'clock" or "not detected").
    Buttons: GPIO 20 DOOR   21 STAIRS   16 ELEVATOR

OBSTACLE DETECTION (continuous, background threads)
    Three sensors each write into a shared state under a lock; a fusion thread
    combines them and drives the five zone LEDs. All of this keeps running while
    a detection + audio sequence plays.
      - LiDAR (YDLiDAR GS)        -> all five zones
      - ToF   (VL53L5CX 8x8)      -> mid_left / center / mid_right
      - Ultrasonic (HC-SR04)      -> center (confirmation)
    Zone LEDs: 17 far-left  23 mid-left  22 center  24 mid-right  27 far-right

MOTION / IDLE (background thread, MPU-6050 "GY-521" on I2C 0x68)
    After IDLE_TIMEOUT_S with no movement the obstacle side goes to sleep: the
    LiDAR motor is turned off, the other sensors stop polling and the zone LEDs
    go dark. Any movement -- or any button press -- wakes it back up. The
    buttons and camera keep working the whole time.

BATTERY (background thread, ADS1115 16-bit ADC on I2C 0x48)
    Reads a 2S LiPo through a resistor divider, converts to percent with a
    resting-voltage curve, and drives GPIO 25 as the low-battery line:
    steady on at or below 20 %, blinking at or below 5 %.

Object detection uses gpiozero (buttons); the obstacle side uses lgpio (LEDs +
ultrasonic), exactly as in ObstacleDetection.py. Pins don't overlap.

I2C NOTE: the ToF (0x29), the MPU-6050 (0x68) and the ADS1115 (0x48) all share
bus 1. Each kernel transfer is atomic, so separate handles are safe, but the
8x8 ToF at 15 Hz already uses a lot of the bus. If truncated-read errors
appear, raise dtparam=i2c_arm_baudrate in /boot/firmware/config.txt (400000)
before lowering TOF_RANGING_FREQ_HZ.
"""

import sys
import time
import os
import glob
import statistics
import struct
import threading

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
from gpiozero import Button, DigitalOutputDevice

try:
    import serial
except ImportError:
    serial = None   # audio optional

try:
    import lgpio
except ImportError:
    lgpio = None    # obstacle LEDs + ultrasonic need this

try:
    import ydlidar
except ImportError:
    ydlidar = None  # lidar sensor optional

try:
    from vl53l5cx.vl53l5cx import VL53L5CX, VL53L5CX_RESOLUTION_8X8
except ImportError:
    VL53L5CX = None  # ToF sensor optional

try:
    from smbus2 import SMBus   # pip install smbus2 (and enable I2C: sudo raspi-config)
except ImportError:
    SMBus = None    # MPU-6050 + ADS1115 need this


# ================== OBJECT DETECTION SETTINGS ==================
MODEL_PATH  = "door_stair_elevator.pt"   # combined model: door, stair, elevator
CONF        = 0.65
SAVE_IMAGES = True
SAVE_DIR    = "detections"

DOOR_BUTTON_PIN     = 20
STAIR_BUTTON_PIN    = 21
ELEVATOR_BUTTON_PIN = 16

DOOR_NAMES     = {"door"}
STAIR_NAMES    = {"stair", "stairs"}
ELEVATOR_NAMES = {"elevator"}

# ================== DFPLAYER (AUDIO) SETTINGS ==================
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 115200
VOLUME      = 18
MIN_VOL, MAX_VOL = 0, 30
CLIP_GAP    = 2.0

# DFPlayer PRO. PLAYMODE: 1=repeat one, 2=repeat all, 3=play one then stop,
# 4=random. PROMPT=OFF silences the beep. Both settings stick on the module.
PLAY_MODE   = 3
PROMPT_TONE = False

CLIP_OBJECT       = {"DOOR": 1, "ELEVATOR": 2, "STAIRS": 3}
CLIP_DETECTED_AT  = {10: 5, 11: 6, 12: 7, 1: 8, 2: 9}
CLIP_NOT_DETECTED = 10
# Spoken low-battery warning. Record a clip, copy it to the module, and set
# this to its file number to enable it. None = warn on the console only.
CLIP_LOW_BATTERY  = None
CLIP_LABEL = {
    1: "Door.mp3", 2: "Elevator.mp3", 3: "Stairs.mp3",
    5: "Detected 10.mp3  (detected, 10 o'clock)",
    6: "Detected 11.mp3  (detected, 11 o'clock)",
    7: "Detected 12.mp3  (detected, 12 o'clock)",
    8: "Detect 1.mp3  (detected, 1 o'clock)",
    9: "Detect at 2.mp3  (detected, 2 o'clock)",
    10: "Not detected.mp3",
}
CLOCK_HOURS = [10, 11, 12, 1, 2]

# ================== OBSTACLE DETECTION SETTINGS ==================
# LiDAR
LIDAR_PORT        = "/dev/ttyUSB0"
LIDAR_BAUD        = 921600
LIDAR_THRESHOLD_M = 1.0
# ToF
TOF_THRESHOLD_MM  = 1000
TOF_MAX_VALID_MM  = 4000
# 8x8 maxes out at 15 Hz. If truncated-read errors persist after raising the
# I2C baudrate in config.txt, drop this to 10 for more read margin.
TOF_RANGING_FREQ_HZ = 15
# Ultrasonic
ULTRASONIC_TRIG          = 19
ULTRASONIC_ECHO          = 13
ULTRASONIC_THRESHOLD_M   = 1.0
ULTRASONIC_MAX_DISTANCE_M = 4.0
ULTRASONIC_MIN_VALID_M   = 0.04
# The 40-pin header's gpiochip number varies: current Raspberry Pi OS on a
# Pi 5 uses 0, older images used 4. Confirm with the `gpiodetect` command.
# (15 did not exist on this Pi; _open_gpiochip() also falls back automatically.)
GPIO_CHIP = 0
# Zone LEDs
LED_PINS = {
    "far_left":  17,
    "mid_left":  23,
    "center":    22,
    "mid_right": 24,
    "far_right": 27,
}

VERBOSE_OBSTACLE = True   # print a line when the fused obstacle zones change

# ================== MOTION / IDLE SETTINGS (MPU-6050) ==================
# Wiring: GY-521 VCC -> 3.3V, GND -> GND, SDA -> GPIO2 (pin 3), SCL -> GPIO3 (pin 5)
IDLE_ENABLED     = True     # False = sensors always on, MPU only reports motion
I2C_BUS          = 1
MPU_ADDR         = 0x68
IDLE_TIMEOUT_S   = 5.0      # still this long -> idle (raise it if it naps mid-walk)
ACCEL_THRESH_G   = 0.15     # change between samples (tune)
GYRO_THRESH_DPS  = 4.0      # deg/s above resting bias (tune)
LIDAR_SETTLE_S   = 1.5      # ignore the MPU while the LiDAR motor spins up/down
MOTION_INTERVAL_S = 0.05
DEBUG_MOTION     = False    # True during bring-up: prints live "# motion" numbers

# ================== BATTERY SETTINGS (ADS1115) ==================
# Wiring: ADS1115 VDD -> 5V, GND -> GND, SDA -> GPIO2, SCL -> GPIO3,
#         ADDR -> GND (0x48). Pull-ups stay on the Pi's 3.3V rail -- the
#         ADS1115 is open-drain on SDA/SCL, so never pull them to 5V.
#   Divider: Battery + -> 10k -> [junction] -> 10k -> GND ; junction -> A0
#            (optional 0.1 uF from the junction to GND)
#   Battery - must share GND with the Pi (the buck converter's GND does this).
BATTERY_PIN       = 25      # low-battery output line (digital; see BatteryMonitor)
ADS_ADDR          = 0x48
ADS_CHANNEL       = 0       # A0
ADS_LSB_V         = 0.0001875   # volts per count at GAIN_TWOTHIRDS (+/-6.144 V)
CELL_COUNT        = 2       # 2S LiPo
DIVIDER_RATIO     = 2.0     # (R1+R2)/R2 -- change if you use other resistors
ADS_NUM_SAMPLES   = 8       # averaged readings
BATTERY_INTERVAL_S = 0.5    # how often to sample the pack
LOW_ON_PCT        = 20.0    # alert at or below this
LOW_OFF_PCT       = 25.0    # clear the alert above this (hysteresis)
CRITICAL_PCT      = 5.0
CRITICAL_BLINK_HZ = 4.0     # GPIO 25 blinks this fast once critical
LOW_REPEAT_S      = 120.0   # min gap between spoken low-battery warnings
BATTERY_SIM       = False   # True = fake a pack draining 8.4 V -> 6.6 V (no ADC)
SIM_START_V       = 8.40
SIM_END_V         = 6.60
SIM_DURATION_S    = 60.0
DEBUG_BATTERY     = True    # periodic "# batt" status line
DEBUG_BATTERY_INTERVAL_S = 10.0

# Per-cell voltage -> percent (resting voltage, approximate)
BATTERY_CURVE = [
    (4.20, 100.0), (4.06, 90.0), (3.98, 80.0), (3.92, 70.0), (3.87, 60.0),
    (3.82, 50.0), (3.79, 40.0), (3.77, 30.0), (3.74, 20.0), (3.68, 10.0),
    (3.45, 5.0), (3.30, 0.0),
]

# ---- shared state across the obstacle threads ----
state_lock = threading.Lock()
shared_state = {
    "lidar": {"far_left": False, "mid_left": False, "center": False, "mid_right": False, "far_right": False},
    "tof_1": {"mid_left": False, "center": False, "mid_right": False},
    "ultrasonic_detected": False,
}
stop_event = threading.Event()

# Set by the motion monitor; the sensor + fusion threads watch it.
idle_event = threading.Event()
# The MotionMonitor instance, so any thread can report a self-inflicted
# disturbance (LiDAR motor) or real activity (button press). None if absent.
motion_monitor = None

# The DFPlayer is one serial port shared by the main thread and the battery
# thread, so every exchange with it goes through this lock.
audio_lock = threading.Lock()

CLEAR_LIDAR_ZONES = {"far_left": False, "mid_left": False, "center": False,
                     "mid_right": False, "far_right": False}
CLEAR_TOF_ZONES = {"mid_left": False, "center": False, "mid_right": False}


def notify_disturbance():
    """LiDAR motor just started or stopped: tell the MPU to ignore it."""
    if motion_monitor is not None:
        motion_monitor.notify_disturbance()


def notify_activity():
    """Count something as movement even if the MPU didn't see it (e.g. a button)."""
    if motion_monitor is not None:
        motion_monitor.notify_activity()


# ----------------------------------------------------------------------
# audio module (DFPlayer)
# ----------------------------------------------------------------------
def send(ser, command, wait=0.25):
    ser.reset_input_buffer()
    ser.write((command + "\r\n").encode("ascii"))
    time.sleep(wait)
    return ser.read_all().decode("ascii", errors="ignore").strip()


def open_audio():
    if serial is None:
        print("Missing 'serial' - running WITHOUT audio.")
        return None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open {SERIAL_PORT}: {e} - running WITHOUT audio.")
        return None

    time.sleep(0.5)
    if "OK" not in send(ser, "AT").upper():
        print("No OK from the DFPlayer - running WITHOUT audio. Check wiring.")
        ser.close()
        return None

    send(ser, f"AT+PROMPT={'ON' if PROMPT_TONE else 'OFF'}")   # kill the beep first
    send(ser, f"AT+VOL={max(MIN_VOL, min(MAX_VOL, VOLUME))}")
    if PLAY_MODE is not None:
        send(ser, f"AT+PLAYMODE={PLAY_MODE}")
    print(f"DFPlayer ready. Volume: {VOLUME}/30")
    return ser


def play_clip(ser, n):
    if n is None:
        return
    print(f"    > play {n:2d}  {CLIP_LABEL.get(n, f'file {n}')}")
    if ser is None:
        return
    with audio_lock:          # keeps the battery thread from cutting in mid-clip
        send(ser, f"AT+PLAYNUM={n}")
        time.sleep(CLIP_GAP)


def play_sequence(ser, numbers):
    for n in numbers:
        play_clip(ser, n)


# ----------------------------------------------------------------------
# obstacle sensors (each runs on its own thread, writes shared_state)
# ----------------------------------------------------------------------
def lidar_worker():
    ydlidar.os_init()
    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, LIDAR_PORT)
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, LIDAR_BAUD)
    laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_GS)
    laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
    laser.setlidaropt(ydlidar.LidarPropSampleRate, 4)
    laser.setlidaropt(ydlidar.LidarPropAbnormalCheckCount, 4)
    laser.setlidaropt(ydlidar.LidarPropIntenstiyBit, 8)
    laser.setlidaropt(ydlidar.LidarPropFixedResolution, False)
    laser.setlidaropt(ydlidar.LidarPropReversion, False)
    laser.setlidaropt(ydlidar.LidarPropInverted, False)
    laser.setlidaropt(ydlidar.LidarPropAutoReconnect, True)
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, False)
    laser.setlidaropt(ydlidar.LidarPropIntenstiy, True)
    laser.setlidaropt(ydlidar.LidarPropSupportMotorDtrCtrl, True)
    laser.setlidaropt(ydlidar.LidarPropSupportHeartBeat, False)
    laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
    laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
    laser.setlidaropt(ydlidar.LidarPropMaxRange, 1.0)
    laser.setlidaropt(ydlidar.LidarPropMinRange, 0.07)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency, 8.0)

    if not laser.initialize():
        print("LiDAR failed to initialize:", laser.DescribeError())
        return
    if not laser.turnOn():
        print("LiDAR failed to turn on:", laser.DescribeError())
        return
    lidar_on = True

    scan = ydlidar.LaserScan()
    while not stop_event.is_set():
        # ---- idle: stop the motor, report nothing ----
        if idle_event.is_set():
            if lidar_on:
                laser.turnOff()
                lidar_on = False
                with state_lock:
                    shared_state["lidar"] = CLEAR_LIDAR_ZONES.copy()
                notify_disturbance()      # spin-down isn't "motion"
            stop_event.wait(0.1)
            continue

        # ---- woke up: restart the motor ----
        if not lidar_on:
            if laser.turnOn():
                lidar_on = True
                notify_disturbance()      # spin-up isn't "motion" either
            else:
                print("LiDAR failed to turn on:", laser.DescribeError())
                stop_event.wait(1.0)
                continue

        zones = {"far_left": False, "mid_left": False, "center": False, "mid_right": False, "far_right": False}
        if laser.doProcessSimple(scan):
            for point in scan.points:
                distance_m = point.range
                angle_deg = point.angle * 180.0 / 3.14159265
                if 0 < distance_m <= LIDAR_THRESHOLD_M:
                    if angle_deg < -40:
                        zones["far_left"] = True
                    elif -40 <= angle_deg < -20:
                        zones["mid_left"] = True
                    elif -20 <= angle_deg <= 20:
                        zones["center"] = True
                    elif 20 < angle_deg <= 40:
                        zones["mid_right"] = True
                    else:
                        zones["far_right"] = True

        with state_lock:
            shared_state["lidar"] = zones
        time.sleep(0.01)

    laser.turnOff()
    laser.disconnecting()


def tof_worker():
    driver = VL53L5CX()
    if not driver.is_alive():
        print("ToF sensor 1 not alive")
        return

    driver.init()
    driver.set_resolution(VL53L5CX_RESOLUTION_8X8)
    driver.set_ranging_frequency_hz(TOF_RANGING_FREQ_HZ)
    driver.start_ranging()

    def is_valid(status, distance):
        return status != 255 and distance < TOF_MAX_VALID_MM

    while not stop_event.is_set():
        # Idle: keep ranging (restarting it is slow) but stop reading, which
        # frees up the I2C bus and keeps stale zones out of the fusion.
        if idle_event.is_set():
            with state_lock:
                shared_state["tof_1"] = CLEAR_TOF_ZONES.copy()
            stop_event.wait(0.1)
            continue

        zones = {"mid_left": False, "center": False, "mid_right": False}
        if driver.check_data_ready():
            try:
                ranging_data = driver.get_ranging_data()
            except (IndexError, OSError):
                # truncated/short I2C read - skip this frame, keep thread alive
                time.sleep(0.01)
                continue
            for i in range(64):
                status = ranging_data.target_status[driver.nb_target_per_zone * i]
                distance = ranging_data.distance_mm[driver.nb_target_per_zone * i]
                if is_valid(status, distance) and distance <= TOF_THRESHOLD_MM:
                    col = i % 8
                    if col in (0, 1):
                        zones["mid_left"] = True
                    elif col in (2, 3, 4, 5):
                        zones["center"] = True
                    elif col in (6, 7):
                        zones["mid_right"] = True

            with state_lock:
                shared_state["tof_1"] = zones
        time.sleep(0.01)

    driver.stop_ranging()


def _open_gpiochip():
    """Open the 40-pin header's gpiochip. Its number varies by OS/board, so try
    the configured GPIO_CHIP first, then the common Pi values, and raise a clear
    error if none open."""
    tried = []
    for chip in (GPIO_CHIP, 0, 4):
        if chip in tried:
            continue
        tried.append(chip)
        try:
            return lgpio.gpiochip_open(chip)
        except Exception:
            pass
    raise RuntimeError(f"could not open a gpiochip (tried {tried}); "
                       f"run 'gpiodetect' and set GPIO_CHIP")


def ultrasonic_worker():
    h = _open_gpiochip()
    lgpio.gpio_claim_output(h, ULTRASONIC_TRIG)
    lgpio.gpio_write(h, ULTRASONIC_TRIG, 0)
    lgpio.gpio_claim_alert(h, ULTRASONIC_ECHO, lgpio.BOTH_EDGES)

    edges = {"rise": None, "fall": None}
    done = threading.Event()

    def cb(chip, gpio, level, tick):
        if level == 1:
            edges["rise"] = tick
        elif level == 0 and edges["rise"] is not None:
            edges["fall"] = tick
            done.set()

    lgpio.callback(h, ULTRASONIC_ECHO, lgpio.BOTH_EDGES, cb)
    time.sleep(0.5)

    def measure():
        edges["rise"] = edges["fall"] = None
        done.clear()
        lgpio.gpio_write(h, ULTRASONIC_TRIG, 1)
        time.sleep(0.00001)
        lgpio.gpio_write(h, ULTRASONIC_TRIG, 0)
        if not done.wait(0.03):
            return None
        d = (edges["fall"] - edges["rise"]) * 1e-9 * 343 / 2
        return d if ULTRASONIC_MIN_VALID_M <= d <= ULTRASONIC_MAX_DISTANCE_M else None

    hits = 0
    try:
        while not stop_event.is_set():
            # Idle: stop pinging (the transducer is the noisiest thing here).
            if idle_event.is_set():
                hits = 0
                with state_lock:
                    shared_state["ultrasonic_detected"] = False
                stop_event.wait(0.1)
                continue

            readings = []
            for _ in range(3):
                d = measure()
                if d is not None:
                    readings.append(d)
                time.sleep(0.03)

            close = bool(readings) and statistics.median(readings) <= ULTRASONIC_THRESHOLD_M
            hits = hits + 1 if close else 0   # 2 consecutive hits = obstacle
            with state_lock:
                shared_state["ultrasonic_detected"] = hits >= 2
    finally:
        lgpio.gpiochip_close(h)


def fusion_worker():
    """Combine the three sensors and drive the zone LEDs."""
    h = _open_gpiochip()
    for pin in LED_PINS.values():
        lgpio.gpio_claim_output(h, pin, 0)

    def set_led(zone, on):
        lgpio.gpio_write(h, LED_PINS[zone], 1 if on else 0)

    prev = None
    was_idle = False
    try:
        while not stop_event.is_set():
            # ---- idle: LEDs dark, nothing to fuse ----
            if idle_event.is_set():
                if not was_idle:
                    was_idle = True
                    prev = None
                    for zone in LED_PINS:
                        set_led(zone, False)
                    print("[idle] no movement - obstacle sensors + LEDs OFF")
                stop_event.wait(0.1)
                continue
            if was_idle:
                was_idle = False
                print("[wake] movement detected - obstacle sensors ON")

            with state_lock:
                lidar = shared_state["lidar"].copy()
                tof_1 = shared_state["tof_1"].copy()
                ultrasonic = shared_state["ultrasonic_detected"]

            far_left  = lidar["far_left"]
            far_right = lidar["far_right"]
            mid_left  = lidar["mid_left"]  or tof_1["mid_left"]
            mid_right = lidar["mid_right"] or tof_1["mid_right"]
            center    = lidar["center"]    or tof_1["center"]
            if not lidar["center"] and ultrasonic:
                center = True

            fused = {"far_left": far_left, "mid_left": mid_left, "center": center,
                     "mid_right": mid_right, "far_right": far_right}
            for zone, on in fused.items():
                set_led(zone, on)

            if VERBOSE_OBSTACLE and fused != prev:
                active = [z for z, v in fused.items() if v]
                print(f"[obstacle] {', '.join(active) if active else 'clear'}")
                prev = fused

            time.sleep(0.02)
    finally:
        stop_event.set()
        for zone in LED_PINS:
            try:
                set_led(zone, False)
            except Exception:
                pass
        lgpio.gpiochip_close(h)


def start_obstacle_threads():
    """Start whatever obstacle sensors are available. Returns the thread list."""
    if lgpio is None:
        print("Obstacle detection OFF - 'lgpio' not installed.")
        return []

    threads = []
    if ydlidar is not None:
        threads.append(threading.Thread(target=lidar_worker, daemon=True))
    else:
        print("LiDAR: 'ydlidar' not installed - skipping that sensor.")
    if VL53L5CX is not None:
        threads.append(threading.Thread(target=tof_worker, daemon=True))
    else:
        print("ToF: 'vl53l5cx' not installed - skipping that sensor.")
    threads.append(threading.Thread(target=ultrasonic_worker, daemon=True))
    threads.append(threading.Thread(target=fusion_worker, daemon=True))

    for t in threads:
        t.start()
    return threads


# ----------------------------------------------------------------------
# motion / idle (MPU-6050)
# ----------------------------------------------------------------------
class MotionMonitor:
    """
    Polls an MPU-6050 in a background thread and sets `idle_event` after
    IDLE_TIMEOUT_S without motion; any motion clears it again.

    Motion = change in acceleration between samples, or gyro rate above its
    resting bias. Samples are skipped while the LiDAR motor is changing speed
    and a fresh baseline is taken afterwards, so the cane shaking itself isn't
    mistaken for the user walking.
    """

    def __init__(self):
        self._bus = SMBus(I2C_BUS)
        self._bus.write_byte_data(MPU_ADDR, 0x6B, 0)   # wake it (starts asleep)
        time.sleep(0.1)

        self._gyro_bias = self._calibrate_gyro()
        first = self._read()
        self._prev_accel = first[0] if first else (0.0, 0.0, 0.0)

        self._need_baseline = True
        self._blank_until = 0.0
        self._last_motion = time.monotonic()
        self._d_accel = 0.0
        self._gyro_mag = 0.0

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def idle(self):
        return idle_event.is_set()

    @property
    def still_for(self):
        return time.monotonic() - self._last_motion

    def notify_disturbance(self):
        """Call after the LiDAR motor starts/stops: ignore the MPU briefly."""
        self._blank_until = time.monotonic() + LIDAR_SETTLE_S
        self._need_baseline = True

    def notify_activity(self):
        """Treat something else (a button press) as movement."""
        self._last_motion = time.monotonic()
        idle_event.clear()

    def _read(self):
        try:
            data = self._bus.read_i2c_block_data(MPU_ADDR, 0x3B, 14)
        except OSError:
            return None
        # ax, ay, az, temp, gx, gy, gz  (big-endian signed 16-bit)
        ax, ay, az, _, gx, gy, gz = struct.unpack(">hhhhhhh", bytes(data))
        accel = (ax / 16384.0, ay / 16384.0, az / 16384.0)   # g (+/-2g)
        gyro = (gx / 131.0, gy / 131.0, gz / 131.0)          # deg/s (+/-250)
        return accel, gyro

    def _calibrate_gyro(self):
        """Keep the device still at startup."""
        sums = [0.0, 0.0, 0.0]
        n = 0
        for _ in range(40):
            r = self._read()
            if r:
                for i in range(3):
                    sums[i] += r[1][i]
                n += 1
            time.sleep(0.025)
        return tuple(s / n for s in sums) if n else (0.0, 0.0, 0.0)

    def _motion_detected(self):
        r = self._read()
        if r is None:
            return False
        accel, gyro = r
        self._d_accel = sum(abs(a - p) for a, p in zip(accel, self._prev_accel))
        self._prev_accel = accel
        self._gyro_mag = sum(abs(g - b) for g, b in zip(gyro, self._gyro_bias))
        return self._d_accel > ACCEL_THRESH_G or self._gyro_mag > GYRO_THRESH_DPS

    def _run(self):
        last_debug = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            quiet = now >= self._blank_until

            if not quiet:
                self._need_baseline = True      # readings now are polluted
            elif self._need_baseline:
                r = self._read()                # first quiet sample: baseline only
                if r:
                    self._prev_accel = r[0]
                self._need_baseline = False
            elif self._motion_detected():
                self._last_motion = now
                idle_event.clear()

            if DEBUG_MOTION and now - last_debug >= 0.2:
                last_debug = now
                print(f"# motion dAccel={self._d_accel:.3f} (thr {ACCEL_THRESH_G:.2f}) "
                      f"gyro={self._gyro_mag:.2f} (thr {GYRO_THRESH_DPS:.1f}) "
                      f"quiet={int(quiet)} stillFor={now - self._last_motion:.1f}s")

            if IDLE_ENABLED and now - self._last_motion >= IDLE_TIMEOUT_S:
                idle_event.set()

            self._stop_event.wait(MOTION_INTERVAL_S)

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        idle_event.clear()
        self._bus.close()


def start_motion_monitor():
    """Returns a MotionMonitor, or None if the MPU-6050 isn't usable."""
    global motion_monitor
    if SMBus is None:
        print("Motion/idle OFF - 'smbus2' not installed (sensors stay always-on).")
        return None
    try:
        motion_monitor = MotionMonitor()
    except OSError as e:
        print(f"Motion/idle OFF - MPU-6050 not found at 0x{MPU_ADDR:02x}: {e}")
        return None
    print(f"MPU-6050 ready. Idle after {IDLE_TIMEOUT_S:.0f}s still"
          if IDLE_ENABLED else "MPU-6050 ready. Idle sleep disabled")
    return motion_monitor


# ----------------------------------------------------------------------
# battery (ADS1115)
# ----------------------------------------------------------------------
def cell_voltage_to_percent(v):
    """Interpolate the resting-voltage curve (same table as the .ino)."""
    if v >= BATTERY_CURVE[0][0]:
        return 100.0
    if v <= BATTERY_CURVE[-1][0]:
        return 0.0
    for (v_hi, pct_hi), (v_lo, pct_lo) in zip(BATTERY_CURVE, BATTERY_CURVE[1:]):
        if v_lo <= v <= v_hi:
            frac = (v - v_lo) / (v_hi - v_lo)
            return pct_lo + frac * (pct_hi - pct_lo)
    return 0.0


class BatteryMonitor:
    """
    Reads the 2S LiPo through an ADS1115 and reports percent remaining, with
    the .ino's 20 % / 25 % hysteresis.

    GPIO 25 is a digital pin, so it can't carry a percentage directly -- it is
    the low-battery line (the equivalent of LOW_LED_PIN in the .ino):

        battery > 25 %   -> low
        battery <= 20 %  -> high (steady)
        battery <= 5 %   -> blinking at CRITICAL_BLINK_HZ

    The number itself is on `.percent`, for logging or a display. If GPIO 25
    has to carry the value rather than a flag, see the note at the end of the
    class.
    """

    def __init__(self, pin=BATTERY_PIN, ser=None):
        self._device = DigitalOutputDevice(pin)
        self._ser = ser
        self._bus = None
        self._ads_ok = False
        self._percent = None
        self._pack_v = 0.0
        self._raw = -1
        self._low = False
        self._critical = False
        self._last_warn = 0.0
        self._start = time.monotonic()

        if not BATTERY_SIM:
            if SMBus is None:
                print("Battery monitor OFF - 'smbus2' not installed.")
            else:
                try:
                    self._bus = SMBus(I2C_BUS)
                    self._read_raw()      # probe: raises OSError if nothing at 0x48
                    self._ads_ok = True
                except OSError as e:
                    print(f"# ERROR: ADS1115 not found at 0x{ADS_ADDR:02x} "
                          f"- check wiring/ADDR pin ({e})")

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def percent(self):
        """Percent remaining, or None if the pack hasn't been read yet."""
        return self._percent

    @property
    def low(self):
        return self._low

    @property
    def critical(self):
        return self._critical

    def _read_raw(self):
        """One single-shot conversion on ADS_CHANNEL at GAIN_TWOTHIRDS."""
        # OS=1 start, MUX=100+ch (AINx vs GND), PGA=000 (+/-6.144 V),
        # MODE=1 single-shot, DR=100 (128 SPS), comparator disabled.
        config = 0x8183 | ((0x04 | ADS_CHANNEL) << 12)
        self._bus.write_i2c_block_data(ADS_ADDR, 0x01,
                                       [(config >> 8) & 0xFF, config & 0xFF])
        time.sleep(0.009)      # 128 SPS -> ~7.8 ms conversion
        data = self._bus.read_i2c_block_data(ADS_ADDR, 0x00, 2)
        return struct.unpack(">h", bytes(data))[0]

    def _read_pack_voltage(self):
        """Returns (pack_volts, adc_raw); raw is -1 if the ADC couldn't be read."""
        if BATTERY_SIM:
            t = (time.monotonic() - self._start) % (SIM_DURATION_S + 3.0)
            frac = min(t, SIM_DURATION_S) / SIM_DURATION_S
            v = SIM_START_V - frac * (SIM_START_V - SIM_END_V)
            return v, int((v / DIVIDER_RATIO) / ADS_LSB_V)
        if not self._ads_ok:
            return 0.0, -1
        total = 0
        try:
            for _ in range(ADS_NUM_SAMPLES):
                total += self._read_raw()
        except OSError:
            return 0.0, -1
        raw = total // ADS_NUM_SAMPLES
        return raw * ADS_LSB_V * DIVIDER_RATIO, raw

    def _warn(self, text):
        print(text)
        now = time.monotonic()
        if CLIP_LOW_BATTERY is not None and now - self._last_warn >= LOW_REPEAT_S:
            self._last_warn = now
            play_clip(self._ser, CLIP_LOW_BATTERY)   # audio_lock keeps this polite

    def _update_flags(self, pct):
        if not self._low and pct <= LOW_ON_PCT:
            self._low = True
            self._warn("[battery] LOW: 20% or less remaining")
        elif self._low and pct > LOW_OFF_PCT:
            self._low = False
            self._critical = False
            print("[battery] recovered above 25%")
        if pct <= CRITICAL_PCT and not self._critical:
            self._critical = True
            self._warn("[battery] CRITICAL: nearly empty, stop and recharge")

    def _drive_pin(self, now):
        if self._critical:
            period = 1.0 / CRITICAL_BLINK_HZ
            self._device.value = (now % period) < period / 2
        else:
            self._device.value = self._low

    def _run(self):
        last_sample = 0.0
        last_debug = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - last_sample >= BATTERY_INTERVAL_S:
                last_sample = now
                pack_v, raw = self._read_pack_voltage()
                self._raw = raw
                if raw >= 0:
                    self._pack_v = pack_v
                    cell_v = pack_v / CELL_COUNT      # assumes balanced cells
                    self._percent = cell_voltage_to_percent(cell_v)
                    self._update_flags(self._percent)
                    if DEBUG_BATTERY and now - last_debug >= DEBUG_BATTERY_INTERVAL_S:
                        last_debug = now
                        print(f"# batt raw={raw} pack={pack_v:.3f}V "
                              f"cell={cell_v:.3f}V pct={self._percent:.1f} "
                              f"low={int(self._low)}")
            self._drive_pin(now)
            self._stop_event.wait(0.02)   # fine enough for the critical blink

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._device.off()
        self._device.close()
        if self._bus is not None:
            self._bus.close()

    # If GPIO 25 must carry the percentage itself rather than a low flag, swap
    # DigitalOutputDevice for gpiozero's PWMOutputDevice(BATTERY_PIN) and set
    # `self._device.value = (self._percent or 0) / 100.0` in _drive_pin(). That
    # gives a duty cycle proportional to charge, readable by a scope, an RC
    # filter into another ADC, or an LED whose brightness tracks the pack.


# ----------------------------------------------------------------------
# object detection
# ----------------------------------------------------------------------
# make a filename that doesn't already exist so I don't overwrite old ones
def next_save_path(directory, label):
    os.makedirs(directory, exist_ok=True)
    existing = glob.glob(os.path.join(directory, f"{label}_*.jpg"))
    nums = []
    for f in existing:
        name = os.path.splitext(os.path.basename(f))[0]
        suffix = name.replace(f"{label}_", "")
        if suffix.isdigit():
            nums.append(int(suffix))
    n = (max(nums) + 1) if nums else 1
    return os.path.join(directory, f"{label}_{n:04d}.jpg")


def clock_position(box, frame_width):
    x1, _, x2, _ = box.xyxy[0].tolist()
    box_center = (x1 + x2) / 2
    x_norm = max(0.0, min(1.0, box_center / frame_width))
    idx = min(len(CLOCK_HOURS) - 1, int(x_norm * len(CLOCK_HOURS)))
    return CLOCK_HOURS[idx]


def detect(picam2, model, ser, target_names, target_label):
    frame = picam2.capture_array()
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_width = frame.shape[1]
    r = model(frame, conf=CONF, verbose=False)[0]

    hits = []
    for box in r.boxes:
        name = model.names[int(box.cls[0])]
        if name.lower() in target_names:
            conf = float(box.conf[0])
            hour = clock_position(box, frame_width)
            hits.append((name, conf, hour))

    print(f"\n--- {target_label} ---")

    if hits:
        hits.sort(key=lambda x: x[1], reverse=True)
        best_name, best_conf, best_hour = hits[0]
        print(f"  DETECTED at {best_hour} o'clock, {round(best_conf*100)}% "
              f"confidence (best match: {best_name})")
        if len(hits) > 1:
            print(f"  ({len(hits)} total detections)")
        play_sequence(ser, [CLIP_OBJECT[target_label], CLIP_DETECTED_AT[best_hour]])
        tag = target_label
    else:
        print(f"  NOT FOUND: no {target_label.lower()} detected.")
        play_sequence(ser, [CLIP_OBJECT[target_label], CLIP_NOT_DETECTED])
        tag = f"{target_label}_NOTFOUND"

    if SAVE_IMAGES:
        out_path = next_save_path(SAVE_DIR, tag)
        cv2.imwrite(out_path, r.plot())
        print(f"  Saved image: {out_path}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Model classes:", model.names)

    print("Starting audio module...")
    ser = open_audio()

    print("Starting camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()
    time.sleep(2)

    door_button     = Button(DOOR_BUTTON_PIN)
    stair_button    = Button(STAIR_BUTTON_PIN)
    elevator_button = Button(ELEVATOR_BUTTON_PIN)

    # Keep the device still for this: it calibrates the gyro's resting bias.
    print("Starting motion sensor (hold still)...")
    monitor = start_motion_monitor()

    print("Starting battery monitor...")
    battery = BatteryMonitor(ser=ser)

    print("Starting obstacle sensors...")
    obstacle_threads = start_obstacle_threads()

    print("\nReady.")
    print(f"  Door button     (GPIO {DOOR_BUTTON_PIN})  -> look for a DOOR")
    print(f"  Stair button    (GPIO {STAIR_BUTTON_PIN})  -> look for STAIRS")
    print(f"  Elevator button (GPIO {ELEVATOR_BUTTON_PIN})  -> look for an ELEVATOR")
    print(f"  Obstacle sensors: {len(obstacle_threads)} thread(s) running")
    print(f"  Motion/idle: {'on' if monitor and IDLE_ENABLED else 'off'}"
          f"   Battery line: GPIO {BATTERY_PIN}")
    print("  Ctrl+C to quit")

    try:
        while not stop_event.is_set():
            pressed = None
            if door_button.is_pressed:
                pressed = (DOOR_NAMES, "DOOR", door_button)
            elif stair_button.is_pressed:
                pressed = (STAIR_NAMES, "STAIRS", stair_button)
            elif elevator_button.is_pressed:
                pressed = (ELEVATOR_NAMES, "ELEVATOR", elevator_button)

            if pressed is not None:
                names, label, button = pressed
                # A button press counts as activity, so the obstacle sensors
                # are already awake by the time the audio finishes.
                notify_activity()
                if battery.low:
                    pct = battery.percent
                    print(f"  (battery {pct:.0f}% - low)" if pct is not None
                          else "  (battery low)")
                detect(picam2, model, ser, names, label)
                notify_activity()
                button.wait_for_release()

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop_event.set()
        for t in obstacle_threads:
            t.join(timeout=2.0)
        battery.close()
        if monitor is not None:
            monitor.close()
        picam2.stop()
        picam2.close()
        if ser is not None:
            ser.close()
        print("Done.")


if __name__ == "__main__":
    main()
