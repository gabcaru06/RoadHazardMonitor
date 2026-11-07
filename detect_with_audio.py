#!/usr/bin/env python3
"""
Button-triggered object detection for the Pi, with spoken results played
through a DFPlayer-style serial MP3 module (headphones on the module).

Three buttons, one per target object (door / stairs / elevator). Press a
button -> the Pi grabs a frame and checks ONLY for that object -> it then
speaks the result by playing a short SEQUENCE of pre-recorded clips off
the module's SD card:
    found:    "<object> detected"  +  "<n> percent"  +  "<n> o'clock"
    not found:"no <object> detected"

Because the module can only play pre-recorded, numbered files, the spoken
sentence is assembled from building-block clips. See CLIP MAP below for
exactly which file number must contain which phrase.
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
    serial = None   # audio is optional - we warn and keep going without it


# --- detection settings ---
# Same weights the batch HEIC script verifies against.
# Model classes are exactly: "door", "stair", "elevator".
MODEL_PATH  = "door_stair_elevator.pt"
CONF        = 0.65           # matches the threshold used in the batch run
SAVE_IMAGES = True
SAVE_DIR    = "detections"

DOOR_BUTTON_PIN     = 2      # button to look for a DOOR
STAIR_BUTTON_PIN    = 3      # button to look for STAIRS
ELEVATOR_BUTTON_PIN = 4      # button to look for an ELEVATOR  (set to your wiring)

# Class names the model actually emits (matched case-insensitively).
DOOR_NAMES     = {"door"}
STAIR_NAMES    = {"stair"}
ELEVATOR_NAMES = {"elevator"}

# --- audio module (DFPlayer-style serial MP3 player) ---
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 115200
VOLUME      = 18             # 0-30 (headphones can be quieter - line level)
MIN_VOL, MAX_VOL = 0, 30

# Seconds to wait for one clip to finish before playing the next.
# Set this a little longer than your LONGEST recorded clip.
CLIP_GAP = 1.5

# ====================== CLIP MAP (what to record) ======================
# Record these 11 clips and put them on the SD card as the matching file
# numbers (e.g. 0001.mp3 ... 0011.mp3, copied in order). Keep each clip
# shorter than CLIP_GAP. (Confidence is NOT spoken.)
#
#   1  "door detected"            7  "ten o'clock"
#   2  "stairs detected"          8  "eleven o'clock"
#   3  "elevator detected"        9  "twelve o'clock"
#   4  "no door detected"        10  "one o'clock"
#   5  "no stairs detected"      11  "two o'clock"
#   6  "no elevator detected"
# =======================================================================
CLIP_DETECTED  = {"DOOR": 1, "STAIRS": 2, "ELEVATOR": 3}
CLIP_NOT_FOUND = {"DOOR": 4, "STAIRS": 5, "ELEVATOR": 6}
CLIP_CLOCK     = {10: 7, 11: 8, 12: 9, 1: 10, 2: 11}

# Clock positions across the frame, left edge -> right edge.
CLOCK_HOURS = [10, 11, 12, 1, 2]


# ----------------------------------------------------------------------
# audio module
# ----------------------------------------------------------------------
def send(ser, command, wait=0.25):
    ser.reset_input_buffer()
    ser.write((command + "\r\n").encode("ascii"))
    time.sleep(wait)
    return ser.read_all().decode("ascii", errors="ignore").strip()


def set_volume(ser, vol):
    vol = max(MIN_VOL, min(MAX_VOL, vol))
    send(ser, f"AT+VOL={vol}")
    return vol


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
        print("No OK from the audio module - running WITHOUT audio. "
              "Check wiring/power.")
        ser.close()
        return None

    # No AT+AMP - the headphone jack runs off the DAC, not the amp.
    # No repeat mode: we want play-once so a clip sequence advances cleanly.
    set_volume(ser, VOLUME)
    print(f"Audio module ready. Volume: {VOLUME}/30")
    return ser


def play_clip(ser, n):
    if ser is None or n is None:
        return
    send(ser, f"AT+PLAYNUM={n}")
    time.sleep(CLIP_GAP)        # let it finish before the next clip


def play_sequence(ser, numbers):
    for n in numbers:
        play_clip(ser, n)


# ----------------------------------------------------------------------
# detection helpers
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

    print(f"\nLooking for: {target_label} ---")

    if hits:
        hits.sort(key=lambda x: x[1], reverse=True)   # best guess first
        best_name, best_conf, best_hour = hits[0]
        print(f"  DETECTED at {best_hour} o'clock, {round(best_conf*100)}% "
              f"confidence (best match: {best_name})")
        if len(hits) > 1:
            print(f"  ({len(hits)} total detections)")
        # spoken: "<object> detected" -> "<n> o'clock"  (confidence not spoken)
        play_sequence(ser, [
            CLIP_DETECTED[target_label],
            CLIP_CLOCK[best_hour],
        ])
        tag = target_label
    else:
        print(f"  NOT FOUND: no {target_label} detected.")
        play_clip(ser, CLIP_NOT_FOUND[target_label])
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
    config = picam2.create_preview_configuration()
    picam2.configure(config)
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
