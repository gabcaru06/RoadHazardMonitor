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

Object detection uses gpiozero (buttons); the obstacle side uses lgpio (LEDs +
ultrasonic), exactly as in ObstacleDetection.py. Pins don't overlap.
"""

import sys
import time
import os
import glob
import statistics
import threading

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
from gpiozero import Button

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

# ---- shared state across the obstacle threads ----
state_lock = threading.Lock()
shared_state = {
    "lidar": {"far_left": False, "mid_left": False, "center": False, "mid_right": False, "far_right": False},
    "tof_1": {"mid_left": False, "center": False, "mid_right": False},
    "ultrasonic_detected": False,
}
stop_event = threading.Event()


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

    scan = ydlidar.LaserScan()
    while not stop_event.is_set():
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
    try:
        while not stop_event.is_set():
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

    print("Starting obstacle sensors...")
    obstacle_threads = start_obstacle_threads()

    print("\nReady.")
    print(f"  Door button     (GPIO {DOOR_BUTTON_PIN})  -> look for a DOOR")
    print(f"  Stair button    (GPIO {STAIR_BUTTON_PIN})  -> look for STAIRS")
    print(f"  Elevator button (GPIO {ELEVATOR_BUTTON_PIN})  -> look for an ELEVATOR")
    print(f"  Obstacle sensors: {len(obstacle_threads)} thread(s) running")
    print("  Ctrl+C to quit")

    try:
        while not stop_event.is_set():
            if door_button.is_pressed:
                detect(picam2, model, ser, DOOR_NAMES, "DOOR")
                door_button.wait_for_release()
            elif stair_button.is_pressed:
                detect(picam2, model, ser, STAIR_NAMES, "STAIRS")
                stair_button.wait_for_release()
            elif elevator_button.is_pressed:
                detect(picam2, model, ser, ELEVATOR_NAMES, "ELEVATOR")
                elevator_button.wait_for_release()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop_event.set()
        for t in obstacle_threads:
            t.join(timeout=2.0)
        picam2.stop()
        picam2.close()
        if ser is not None:
            ser.close()
        print("Done.")


if __name__ == "__main__":
    main()
