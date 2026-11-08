#!/usr/bin/env python3

import sys
import time

try:
    import serial
except ImportError:
    print("Missing 'serial'. Install with:  sudo apt install -y python3-serial")
    sys.exit(1)

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 115200
FILE_NUMBER = 1         # the only file on the module ("never")
START_VOLUME = 18       # 0-30  (headphones can be quieter - line level)
STEP = 3
MIN_VOL, MAX_VOL = 0, 30

MODE_REPEAT_ONE = 1     # loop the single track


def send(ser, command, wait=0.25):
    ser.reset_input_buffer()
    ser.write((command + "\r\n").encode("ascii"))
    time.sleep(wait)
    return ser.read_all().decode("ascii", errors="ignore").strip()


def set_volume(ser, vol):
    vol = max(MIN_VOL, min(MAX_VOL, vol))
    send(ser, f"AT+VOL={vol}")
    return vol


def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open {SERIAL_PORT}: {e}")
        sys.exit(1)

    time.sleep(0.5)

    if "OK" not in send(ser, "AT").upper():
        print("No OK from the module - check wiring/power, then retry.")
        ser.close()
        sys.exit(1)

    # No AT+AMP here - the headphone jack runs off the DAC, not the amp.
    volume = set_volume(ser, START_VOLUME)
    send(ser, f"AT+PLAYMODE={MODE_REPEAT_ONE}")   # loop it
    send(ser, f"AT+PLAYNUM={FILE_NUMBER}")        # play the one file

    print(f"""
Playing through the headphone jack, looping. Volume: {volume}/30

Controls (type, then Enter):
    +        louder  (by {STEP})
    -        quieter (by {STEP})
    0 - 30   jump to that volume
    q        quit
""")

    try:
        while True:
            choice = input(f"[vol {volume}] > ").strip().lower()
            if choice == "q":
                break
            elif choice == "+":
                volume = set_volume(ser, volume + STEP)
            elif choice == "-":
                volume = set_volume(ser, volume - STEP)
            elif choice.isdigit():
                volume = set_volume(ser, int(choice))
            else:
                print("  Use +, -, a number 0-30, or q.")
                continue
            print(f"  volume = {volume}")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
