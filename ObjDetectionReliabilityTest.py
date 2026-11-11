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

HIT_TOLERANCE_MM = 5

TEST_DURATION_MIN = 30
SAMPLE_INTERVAL_S = 0.5

state_lock = threading.Lock()
shared_state = {"lidar_mm": None, "ultrasonic_mm": None}
stop_event = threading.Event()


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
                shared_state["lidar_mm"] = min(valid) if valid else None

        time.sleep(0.01)

    laser.turnOff()
    laser.disconnecting()


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


def is_hit(distance_mm, known_distance_mm, tolerance_mm):
    if distance_mm is None:
        return False
    return abs(distance_mm - known_distance_mm) <= tolerance_mm


if __name__ == "__main__":
    known_distance_m = float(input("Enter the known distance you're placing the obstacle at (in meters): "))
    known_distance_mm = known_distance_m * 1000
    print(f"Using known distance: {known_distance_mm:.1f} mm, hit tolerance: ±{HIT_TOLERANCE_MM} mm")

    threads = [
        threading.Thread(target=lidar_worker, daemon=True),
        threading.Thread(target=ultrasonic_worker, daemon=True),
    ]
    for t in threads:
        t.start()

    print("Letting sensors initialize...")
    time.sleep(2)

    input(f"\nPlace the object at exactly {known_distance_m} m from the sensors, then press Enter to start...")

    total_samples = 0
    hits = {"lidar": 0, "ultrasonic": 0}
    current_miss_streak = {"lidar": 0, "ultrasonic": 0}
    longest_miss_streak = {"lidar": 0, "ultrasonic": 0}

    test_duration_s = TEST_DURATION_MIN * 60
    start_time = time.time()

    print(f"Logging for {TEST_DURATION_MIN} minutes. Do not disturb the object or sensors.")

    try:
        with open("reliability_test.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "known_distance_mm", "lidar_mm", "ultrasonic_mm",
                              "lidar_hit", "ultrasonic_hit"])

            while time.time() - start_time < test_duration_s:
                with state_lock:
                    lidar_mm = shared_state["lidar_mm"]
                    us_mm = shared_state["ultrasonic_mm"]

                elapsed_s = round(time.time() - start_time, 2)

                lidar_hit = is_hit(lidar_mm, known_distance_mm, HIT_TOLERANCE_MM)
                us_hit = is_hit(us_mm, known_distance_mm, HIT_TOLERANCE_MM)

                writer.writerow([elapsed_s, known_distance_mm, lidar_mm, us_mm, lidar_hit, us_hit])

                total_samples += 1
                for name, hit in (("lidar", lidar_hit), ("ultrasonic", us_hit)):
                    if hit:
                        hits[name] += 1
                        current_miss_streak[name] = 0
                    else:
                        current_miss_streak[name] += 1
                        longest_miss_streak[name] = max(longest_miss_streak[name], current_miss_streak[name])

                time.sleep(SAMPLE_INTERVAL_S)

        print("\n=== Reliability Test Results ===")
        print(f"Known distance: {known_distance_mm:.1f} mm (±{HIT_TOLERANCE_MM} mm tolerance)")
        print(f"Total samples: {total_samples}")
        for name in ("lidar", "ultrasonic"):
            hit_rate = (hits[name] / total_samples * 100) if total_samples else 0
            longest_miss_s = longest_miss_streak[name] * SAMPLE_INTERVAL_S
            print(f"{name.upper():12s} hit rate: {hit_rate:5.1f}%  |  longest miss streak: "
                  f"{longest_miss_streak[name]} samples ({longest_miss_s:.1f}s)")

        print("\nSaved to reliability_test.csv")

    finally:
        stop_event.set()
        time.sleep(0.5)
