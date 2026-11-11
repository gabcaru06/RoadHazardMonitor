import time
import csv
import threading
import lgpio
import ydlidar
from vl53l5cx.vl53l5cx import VL53L5CX, VL53L5CX_RESOLUTION_8X8

# ---- CONFIG ----
LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = 921600
LIDAR_ANGLE_TOLERANCE_DEG = 5
LIDAR_THRESHOLD_MM = 1000

TOF_MAX_VALID_MM = 4000
TOF_CENTER_COLS = (2, 3, 4, 5)
TOF_THRESHOLD_MM = 1000

CHIP_NUM = 15
TRIG_LINE = 19
ECHO_LINE = 13
US_TIMEOUT_S = 0.03
US_SETTLE_DELAY_S = 0.0003
US_DEBOUNCE_COUNT = 20
US_THRESHOLD_MM = 1000

NUM_TRIALS = 10
MAX_WAIT_S = 5.0        # give up waiting for a sensor after this long
POLL_INTERVAL_S = 0.005 # how often the main loop checks for a crossing

state_lock = threading.Lock()
shared_state = {"lidar_mm": None, "tof_mm": None, "ultrasonic_mm": None}
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
                shared_state["lidar_mm"] = min(valid) if valid else None

        time.sleep(0.005)

    laser.turnOff()
    laser.disconnecting()


# ---------------------------------------------------------------------------
def tof_worker():
    driver = VL53L5CX()
    if not driver.is_alive():
        print("ToF sensor not alive")
        return

    driver.init()
    driver.set_resolution(VL53L5CX_RESOLUTION_8X8)
    driver.set_ranging_frequency_hz(15)
    driver.start_ranging()

    def is_valid(status, distance):
        return status != 255 and distance < TOF_MAX_VALID_MM

    while not stop_event.is_set():
        if driver.check_data_ready():
            ranging_data = driver.get_ranging_data()
            valid = []
            for i in range(64):
                status = ranging_data.target_status[driver.nb_target_per_zone * i]
                distance = ranging_data.distance_mm[driver.nb_target_per_zone * i]
                col = i % 8
                if col in TOF_CENTER_COLS and is_valid(status, distance):
                    valid.append(distance)

            with state_lock:
                shared_state["tof_mm"] = min(valid) if valid else None

        time.sleep(0.005)

    driver.stop_ranging()


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

        time.sleep(0.03)

    lgpio.gpiochip_close(h)


# ---------------------------------------------------------------------------
def run_trial(trial_num, writer):
    print(f"\nTrial {trial_num}/{NUM_TRIALS}: get ready to move the object into range.")
    input("Press Enter the instant you start moving the object...")

    t_start = time.perf_counter()

    lidar_latency = None
    tof_latency = None
    us_latency = None

    lidar_done = tof_done = us_done = False

    while time.perf_counter() - t_start < MAX_WAIT_S:
        with state_lock:
            lidar_mm = shared_state["lidar_mm"]
            tof_mm = shared_state["tof_mm"]
            us_mm = shared_state["ultrasonic_mm"]

        now = time.perf_counter()

        if not lidar_done and lidar_mm is not None and lidar_mm <= LIDAR_THRESHOLD_MM:
            lidar_latency = (now - t_start) * 1000
            lidar_done = True

        if not tof_done and tof_mm is not None and tof_mm <= TOF_THRESHOLD_MM:
            tof_latency = (now - t_start) * 1000
            tof_done = True

        if not us_done and us_mm is not None and us_mm <= US_THRESHOLD_MM:
            us_latency = (now - t_start) * 1000
            us_done = True

        if lidar_done and tof_done and us_done:
            break

        time.sleep(POLL_INTERVAL_S)

    print(f"  LiDAR: {lidar_latency if lidar_latency else 'no detection'} ms")
    print(f"  ToF: {tof_latency if tof_latency else 'no detection'} ms")
    print(f"  Ultrasonic: {us_latency if us_latency else 'no detection'} ms")

    writer.writerow([trial_num, lidar_latency, tof_latency, us_latency])

    print("Move the object back out of range before the next trial.")
    time.sleep(2)


if __name__ == "__main__":
    threads = [
        threading.Thread(target=lidar_worker, daemon=True),
        threading.Thread(target=tof_worker, daemon=True),
        threading.Thread(target=ultrasonic_worker, daemon=True),
    ]
    for t in threads:
        t.start()

    time.sleep(2)  # let sensors finish init

    try:
        with open("latency_test.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "lidar_latency_ms", "tof_latency_ms", "ultrasonic_latency_ms"])

            for trial in range(1, NUM_TRIALS + 1):
                run_trial(trial, writer)

        print("\nAll latency trials complete! Saved to latency_test.csv")
    finally:
        stop_event.set()
        time.sleep(0.5)
