import time
import csv
import threading
import lgpio
import ydlidar

# ---- CONFIG ----
LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = 921600
LIDAR_ANGLE_TOLERANCE_DEG = 5

CHIP_NUM = 15
TRIG_LINE = 19
ECHO_LINE = 13
US_TIMEOUT_S = 0.03
US_SETTLE_DELAY_S = 0.0003
US_DEBOUNCE_COUNT = 20

SAMPLE_INTERVAL_S = 0.1
TEST_DURATION_S = 10

distances_to_test = ["0.2m", "0.5m", "0.8m", "1.0m", "1.5m", "2.0m"]

state_lock = threading.Lock()
shared_state = {"lidar_mm": None, "ultrasonic_mm": None}
stop_event = threading.Event()


# ---------------------------------------------------------------------------
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
    laser.setlidaropt(ydlidar.LidarPropMaxRange, 2.5)
    laser.setlidaropt(ydlidar.LidarPropMinRange, 0.07)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency, 8.0)

    if not laser.initialize():
        print("LiDAR init failed:", laser.DescribeError())
        return
    if not laser.turnOn():
        print("LiDAR turnOn failed:", laser.DescribeError())
        return

    scan = ydlidar.LaserScan()

    while not stop_event.is_set():
        if laser.doProcessSimple(scan):
            valid = []
            for point in scan.points:
                angle_deg = point.angle * 180.0 / 3.14159265
                distance_mm = point.range * 1000.0
                if abs(angle_deg) <= LIDAR_ANGLE_TOLERANCE_DEG and distance_mm > 0:
                    valid.append(distance_mm)

            with state_lock:
                shared_state["lidar_mm"] = sum(valid) / len(valid) if valid else None

        time.sleep(0.01)

    laser.turnOff()
    laser.disconnecting()


# ---------------------------------------------------------------------------
def ultrasonic_worker():
    h = lgpio.gpiochip_open(CHIP_NUM)
    lgpio.gpio_claim_output(h, TRIG_LINE)
    lgpio.gpio_claim_input(h, ECHO_LINE)
    lgpio.gpio_write(h, TRIG_LINE, 0)
    time.sleep(0.5)

    def wait_for_state(target_state, timeout_s):
        start = time.perf_counter()
        stable_count = 0
        while time.perf_counter() - start < timeout_s:
            if lgpio.gpio_read(h, ECHO_LINE) == target_state:
                stable_count += 1
                if stable_count >= US_DEBOUNCE_COUNT:
                    return time.perf_counter()
            else:
                stable_count = 0
        return None

    while not stop_event.is_set():
        lgpio.gpio_write(h, TRIG_LINE, 1)
        time.sleep(0.00001)
        lgpio.gpio_write(h, TRIG_LINE, 0)
        time.sleep(US_SETTLE_DELAY_S)

        pulse_start = wait_for_state(1, US_TIMEOUT_S)
        distance_mm = None
        if pulse_start is not None:
            pulse_end = wait_for_state(0, US_TIMEOUT_S)
            if pulse_end is not None:
                duration = pulse_end - pulse_start
                distance_mm = (duration * 343 / 2) * 1000

        with state_lock:
            shared_state["ultrasonic_mm"] = distance_mm

        time.sleep(0.05)

    lgpio.gpiochip_close(h)


# ---------------------------------------------------------------------------
def run_test(target_distance, writer):
    print(f"\nPosition target at {target_distance}, centered in front of all sensors.")
    input("Press Enter to start logging...")

    start_time = time.time()
    print(f"Logging for {TEST_DURATION_S} seconds...")

    while time.time() - start_time < TEST_DURATION_S:
        with state_lock:
            lidar_mm = shared_state["lidar_mm"]
            us_mm = shared_state["ultrasonic_mm"]

        timestamp_ms = int((time.time() - start_time) * 1000)
        writer.writerow([timestamp_ms, target_distance, lidar_mm, us_mm])

        time.sleep(SAMPLE_INTERVAL_S)

    print(f"Done with {target_distance}")


if __name__ == "__main__":
    threads = [
        threading.Thread(target=lidar_worker, daemon=True),
        threading.Thread(target=ultrasonic_worker, daemon=True),
    ]
    for t in threads:
        t.start()

    time.sleep(2)  # let sensors finish startup/init before first test

    try:
        with open("combined_distance_accuracy.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "target_distance", "lidar_mm", "ultrasonic_mm"])

            for dist in distances_to_test:
                run_test(dist, writer)

        print("\nAll tests complete! Saved to combined_distance_accuracy.csv")
    finally:
        stop_event.set()
        time.sleep(0.5)
