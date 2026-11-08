#!/usr/bin/env python3
"""
PI PIPELINE + LIDAR OBSTACLE DETECTION  (runs on the Raspberry Pi)

Two things run at once:

1. OBJECT DETECTION (event-driven, on the main thread)
   Three GPIO buttons, one per object. Press a button -> the Pi camera grabs
   a frame -> YOLO checks ONLY for that object -> the result is printed AND
   spoken through the DFPlayer as a two-clip sequence (object, then
   "detected, N o'clock" or "not detected").

2. OBSTACLE DETECTION (continuous, on a background thread)
   The YDLiDAR scans constantly. Any return within OBSTACLE_THRESH_M lights
   the LED for its zone (far-left / mid-left / center / mid-right / far-right).
   This keeps updating even while a detection + audio sequence is playing.

Buttons   (button between pin and GND, internal pull-ups):
    GPIO 20 -> DOOR     GPIO 21 -> STAIRS     GPIO 16 -> ELEVATOR
Zone LEDs:
    GPIO 17 far-left  23 mid-left  22 center  24 mid-right  27 far-right
"""

import sys
import time
import os
import glob
import math
import threading

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
from gpiozero import Button, LED

try:
    import serial
except ImportError:
    serial = None   # audio optional - we warn and keep going without it

try:
    import ydlidar
except ImportError:
    ydlidar = None  # lidar optional - object detection still runs without it


# ---------------- object detection settings ----------------
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

# ---------------- DFPlayer (audio) settings ----------------
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 115200
VOLUME      = 18
MIN_VOL, MAX_VOL = 0, 30
CLIP_GAP    = 2.0          # seconds to let one clip finish before the next

# DFRobot DFPlayer PRO. AT+PLAYMODE: 1=repeat one, 2=repeat all,
# 3=play one then stop, 4=random. The mode sticks, so set it every start.
PLAY_MODE = 3

# SD card file positions (AT+PLAYNUM is 1-based):
#  1 Door   2 Elevator   3 Stairs   5-9 "Detected N o'clock"   10 Not detected
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

# ---------------- LiDAR (obstacle) settings ----------------
LIDAR_PORT        = "/dev/ttyUSB0"
LIDAR_BAUD        = 921600
OBSTACLE_THRESH_M = 1.0     # returns closer than this trigger their zone
LED_PINS   = [17, 23, 22, 24, 27]   # far-left, mid-left, center, mid-right, far-right
ZONE_NAMES = ["far-left", "mid-left", "center", "mid-right", "far-right"]


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
        print("Missing 'serial' - running WITHOUT audio. "
              "Install with: sudo apt install -y python3-serial")
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
# LiDAR obstacle detection (runs on its own thread)
# ----------------------------------------------------------------------
def get_zone(angle_deg):
    if angle_deg < -40:
        return 0
    elif -40 <= angle_deg < -20:
        return 1
    elif -20 <= angle_deg <= 20:
        return 2
    elif 20 < angle_deg <= 40:
        return 3
    else:
        return 4


def init_lidar():
    """Return a turned-on CYdLidar, or None if unavailable / it failed."""
    if ydlidar is None:
        print("LiDAR: 'ydlidar' not installed - obstacle detection OFF.")
        return None

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
        print("LiDAR: initialize failed:", laser.DescribeError())
        return None
    if not laser.turnOn():
        print("LiDAR: turn-on failed:", laser.DescribeError())
        laser.disconnecting()
        return None

    print("LiDAR: scanning.")
    return laser


def lidar_worker(laser, leds, stop_event):
    """Continuously scan; light the zone LED for any close obstacle. Prints a
    one-line summary only when the set of blocked zones changes (no spam)."""
    scan = ydlidar.LaserScan()
    prev = None
    try:
        while not stop_event.is_set() and ydlidar.os_isOk():
            if laser.doProcessSimple(scan):
                zones = [False] * 5
                for point in scan.points:
                    d = point.range
                    if 0 < d <= OBSTACLE_THRESH_M:
                        zones[get_zone(math.degrees(point.angle))] = True

                for led, on in zip(leds, zones):
                    led.on() if on else led.off()

                if zones != prev:
                    active = [ZONE_NAMES[i] for i, z in enumerate(zones) if z]
                    print(f"[lidar] obstacle: {', '.join(active) if active else 'clear'}")
                    prev = zones
            time.sleep(0.05)
    finally:
        for led in leds:
            led.off()
        stop_event.set()   # if the lidar loop ends, tell main to wind down too


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

    # start the LiDAR obstacle thread (if the sensor is available)
    stop_event   = threading.Event()
    laser        = init_lidar()
    leds         = []
    lidar_thread = None
    if laser is not None:
        leds = [LED(p) for p in LED_PINS]
        lidar_thread = threading.Thread(
            target=lidar_worker, args=(laser, leds, stop_event), daemon=True)
        lidar_thread.start()

    print("\nReady.")
    print(f"  Door button     (GPIO {DOOR_BUTTON_PIN})  -> look for a DOOR")
    print(f"  Stair button    (GPIO {STAIR_BUTTON_PIN})  -> look for STAIRS")
    print(f"  Elevator button (GPIO {ELEVATOR_BUTTON_PIN})  -> look for an ELEVATOR")
    print(f"  Obstacle LEDs: {'ON' if laser else 'OFF (no lidar)'}")
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
        if lidar_thread is not None:
            lidar_thread.join(timeout=2.0)
        for led in leds:
            led.off()
        if laser is not None:
            laser.turnOff()
            laser.disconnecting()
        picam2.stop()
        picam2.close()
        if ser is not None:
            ser.close()
        print("Done.")


if __name__ == "__main__":
    main()
