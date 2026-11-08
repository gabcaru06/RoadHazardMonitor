#!/usr/bin/env python3
import os
import sys

import cv2
from ultralytics import YOLO


# --- settings ---
CAMERA_INDEX = 0           # 0 = default webcam (try 1/2 if you have several)
CONF         = 0.50        # detection confidence threshold (lower = more hits)

# Single combined model (still need crosswalk class)
MODEL_PATH  = "door_stair_elevator.pt"
MODEL_PATHS = {
    "DOOR":     MODEL_PATH,
    "ELEVATOR": MODEL_PATH,
    "STAIRS":   MODEL_PATH,
}

# Which class name(s) in a model's output count as each object
TARGET_CLASS = {
    "DOOR":     {"door"},
    "ELEVATOR": {"elevator"},
    "STAIRS":   {"stair", "stairs"},
}

# Keyboard keys that stand in for the three hardware buttons.
KEYS = {ord("d"): "DOOR", ord("e"): "ELEVATOR", ord("s"): "STAIRS"}

# What each thing maps to on the Pi's SD card
# Only printed but will be actually played on DFplayer.
CLIP_OBJECT = {
    "DOOR":     "Door.mp3",
    "ELEVATOR": "Elevator.mp3",
    "STAIRS":   "Stairs.mp3",
}
CLIP_DETECTED_AT = {        # combined "detected, N o'clock" clips
    10: "Detected 10.mp3",
    11: "Detected 11.mp3",
    12: "Detected 12.mp3",
    1:  "Detect 1.mp3",
    2:  "Detect at 2.mp3",
}
CLIP_NOT_DETECTED = "Not detected.mp3"

# Clock positions across the frame, left edge -> right edge.
CLOCK_HOURS = [10, 11, 12, 1, 2]


def load_models():
    """Load each object's model, caching by path so a shared combined model
    is only loaded once. Missing files disable that object's key."""
    cache = {}
    models = {}
    for obj, path in MODEL_PATHS.items():
        if not os.path.isfile(path):
            print(f"  [skip] {obj}: model not found ({path}) - its key is disabled")
            models[obj] = None
            continue
        if path not in cache:
            print(f"  loading {path} ...")
            cache[path] = YOLO(path)
        models[obj] = cache[path]
    return models


def clock_position(x_center, frame_width):
    """Map a horizontal pixel position to a clock hour: 10 = far left,
    12 = straight ahead, 2 = far right."""
    x_norm = max(0.0, min(1.0, x_center / frame_width))
    idx = min(len(CLOCK_HOURS) - 1, int(x_norm * len(CLOCK_HOURS)))
    return CLOCK_HOURS[idx]


def run_detection(model, obj, frame):
    """Detect `obj` in `frame`. Returns (annotated_frame, text, mp3_lines, found)."""
    frame_width = frame.shape[1]
    r = model(frame, conf=CONF, verbose=False)[0]

    hits = []
    if r.boxes is not None:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            if label.lower() in TARGET_CLASS[obj]:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                hour = clock_position((x1 + x2) / 2, frame_width)
                hits.append((conf, hour, (x1, y1, x2, y2)))

    annotated = frame.copy()

    if hits:
        hits.sort(key=lambda h: h[0], reverse=True)   # highest confidence first
        best_conf, best_hour, _ = hits[0]
        for conf, hour, (x1, y1, x2, y2) in hits:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 3)
            cv2.putText(annotated, f"{obj.title()} {conf:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
        text = f"{obj}: {best_hour} o'clock  ({best_conf:.2f})"
        mp3_lines = [
            (CLIP_OBJECT[obj], obj.title()),
            (CLIP_DETECTED_AT[best_hour], f"detected, {best_hour} o'clock"),
        ]
        return annotated, text, mp3_lines, True

    text = f"{obj}: not detected"
    mp3_lines = [
        (CLIP_OBJECT[obj], obj.title()),
        (CLIP_NOT_DETECTED, "not detected"),
    ]
    return annotated, text, mp3_lines, False


def report(obj, text, mp3_lines):
    print(f"\n--- {obj} ---")
    print(f"  {text}")
    print("  would play:")
    for fname, phrase in mp3_lines:
        print(f"    {fname:16s} -> \"{phrase}\"")


def main():
    print("Loading models...")
    models = load_models()
    if all(m is None for m in models.values()):
        print("No models could be loaded - put your .pt files next to this script.")
        sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {CAMERA_INDEX}. Try a different CAMERA_INDEX.")
        sys.exit(1)

    print("\nLive. Click the video window, then press:")
    print("  d = door     e = elevator     s = stairs     q = quit\n")

    window = "Detection test"
    overlay = "d=door  e=elevator  s=stairs  q=quit"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        view = frame.copy()
        cv2.putText(view, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.imshow(window, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in KEYS:
            obj = KEYS[key]
            model = models.get(obj)
            if model is None:
                print(f"\n--- {obj} ---\n  (no model loaded, key disabled)")
                continue

            annotated, text, mp3_lines, found = run_detection(model, obj, frame)
            report(obj, text, mp3_lines)

            # freeze on the annotated result so you can evaluate it, and wait
            # for a key: q quits, anything else goes back to the live feed.
            color = (0, 180, 0) if found else (0, 0, 220)
            cv2.putText(annotated, text, (10, annotated.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(annotated, "PAUSED - any key = resume, q = quit",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(window, annotated)

            if (cv2.waitKey(0) & 0xFF) == ord("q"):   # block here until a key
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()