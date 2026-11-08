#!/usr/bin/env python3
"""
Buttons (internal pull-ups):
    GPIO 20 -> DOOR
    GPIO 21 -> STAIRS
    GPIO 16 -> ELEVATOR  


STILL NEED TO IMPLEMENT CROSSWALKS!!!!!!!    
"""

import sys
import time
import os
import glob

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
from gpiozero import Button

try:
    import serial
except ImportError:
    serial = None   # audio optional - we warn and keep going without it


# Single combined model that detects all three classes: door, stair, elevator.
MODEL_PATH  = "door_stair_elevator.pt"
CONF        = 0.65
SAVE_IMAGES = True
SAVE_DIR    = "detections"

DOOR_BUTTON_PIN     = 20
STAIR_BUTTON_PIN    = 21
ELEVATOR_BUTTON_PIN = 16

# class names in the model's output that count as each object
DOOR_NAMES     = {"door"}
STAIR_NAMES    = {"stair", "stairs"}
ELEVATOR_NAMES = {"elevator"}

# audio module (DFPlayer serial MP3 player)
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 115200
VOLUME      = 18            
MIN_VOL, MAX_VOL = 0, 30
CLIP_GAP    = 2.0          # seconds to let one clip finish before the next


PLAY_MODE = None

# ====================== SD CARD FILE POSITIONS =========================
# AT+PLAYNUM plays by 1-based position on the card.
#  1 Door            7 Detected 12   13 11 o'clock   (unused here)
#  2 Elevator        8 Detect 1      14 12 o'clock   (unused here)
#  3 Stairs          9 Detect at 2   15 1 o'clock    (unused here)
#  4 Crosswalk      10 Not detected  16 2 o'clock    (unused here)
#  5 Detected 10    11 Low battery   17 never        (unused here)
#  6 Detected 11    12 10 o'clock
# "Detected N" / "Detect N" clips are the combined "detected, N o'clock".
# =======================================================================

CLIP_OBJECT       = {"DOOR": 1, "ELEVATOR": 2, "STAIRS": 3}
CLIP_DETECTED_AT  = {10: 5, 11: 6, 12: 7, 1: 8, 2: 9}   # "detected, N o'clock"
CLIP_NOT_DETECTED = 10

#labels for each file position, for the terminal readout
CLIP_LABEL = {
    1: "Door.mp3",
    2: "Elevator.mp3",
    3: "Stairs.mp3",
    5: "Detected 10.mp3  (detected, 10 o'clock)",
    6: "Detected 11.mp3  (detected, 11 o'clock)",
    7: "Detected 12.mp3  (detected, 12 o'clock)",
    8: "Detect 1.mp3  (detected, 1 o'clock)",
    9: "Detect at 2.mp3  (detected, 2 o'clock)",
    10: "Not detected.mp3",
}

# Clock positions across the frame, left edge -> right edge.
CLOCK_HOURS = [10, 11, 12, 1, 2]


# audio module
def send(ser, command, wait=0.25):
    ser.reset_input_buffer()
    ser.write((command + "\r\n").encode("ascii"))
    time.sleep(wait)
    return ser.read_all().decode("ascii", errors="ignore").strip()


def open_audio():
    """Bring up the serial MP3 module. Returns the serial object, or None
    if it isn't there - detection still works (and prints) without audio."""
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
    time.sleep(CLIP_GAP)        # let it finish before the next clip


def play_sequence(ser, numbers):
    for n in numbers:
        play_clip(ser, n)


# detection helpers
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
    """Map the box's horizontal center to a clock hour: 10 at the far
    left, 12 straight ahead, 2 at the far right."""
    x1, _, x2, _ = box.xyxy[0].tolist()
    box_center = (x1 + x2) / 2
    x_norm = max(0.0, min(1.0, box_center / frame_width))   # 0 left .. 1 right
    idx = min(len(CLOCK_HOURS) - 1, int(x_norm * len(CLOCK_HOURS)))
    return CLOCK_HOURS[idx]


def detect(picam2, model, ser, target_names, target_label):
    frame = picam2.capture_array()

    # the camera gives 4 color channels but YOLO wants 3, so drop the extra one
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_width = frame.shape[1]

    results = model(frame, conf=CONF, verbose=False)
    r = results[0]

    # keep only the detections we asked for, with confidence and clock position
    hits = []
    for box in r.boxes:
        name = model.names[int(box.cls[0])]
        if name.lower() in target_names:
            conf = float(box.conf[0])
            hour = clock_position(box, frame_width)
            hits.append((name, conf, hour))

    print(f"\n--- {target_label} ---")

    if hits:
        hits.sort(key=lambda x: x[1], reverse=True)   # best guess first
        best_name, best_conf, best_hour = hits[0]
        print(f"  DETECTED at {best_hour} o'clock, {round(best_conf*100)}% "
              f"confidence (best match: {best_name})")
        if len(hits) > 1:
            print(f"  ({len(hits)} total detections)")
        play_sequence(ser, [
            CLIP_OBJECT[target_label],
            CLIP_DETECTED_AT[best_hour],
        ])
        tag = target_label
    else:
        print(f"  NOT FOUND: no {target_label.lower()} detected.")
        play_sequence(ser, [
            CLIP_OBJECT[target_label],
            CLIP_NOT_DETECTED,
        ])
        tag = f"{target_label}_NOTFOUND"

    if SAVE_IMAGES:
        annotated = r.plot()
        out_path = next_save_path(SAVE_DIR, tag)
        cv2.imwrite(out_path, annotated)
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
    time.sleep(2)  # give it a sec to adjust the exposure

    door_button     = Button(DOOR_BUTTON_PIN)
    stair_button    = Button(STAIR_BUTTON_PIN)
    elevator_button = Button(ELEVATOR_BUTTON_PIN)

    print("\nReady.")
    print(f"  Door button     (GPIO {DOOR_BUTTON_PIN})  -> look for a DOOR")
    print(f"  Stair button    (GPIO {STAIR_BUTTON_PIN})  -> look for STAIRS")
    print(f"  Elevator button (GPIO {ELEVATOR_BUTTON_PIN})  -> look for an ELEVATOR")
    print("  Ctrl+C to quit")

    try:
        while True:
            if door_button.is_pressed:
                detect(picam2, model, ser, DOOR_NAMES, "DOOR")
                door_button.wait_for_release()       # one press = one detection
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
        picam2.stop()
        picam2.close()
        if ser is not None:
            ser.close()
        print("Done.")


if __name__ == "__main__":
    main()
