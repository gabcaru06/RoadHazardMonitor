import threading
import time
import statistics
import subprocess, re
import lgpio
import ydlidar
from vl53l5cx.vl53l5cx import VL53L5CX, VL53L5CX_RESOLUTION_8X8

# Lidar Config
LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = 921600
LIDAR_THRESHOLD_M = 1.0

# ToF Config
TOF_THRESHOLD_MM = 1000
TOF_MAX_VALID_MM = 4000

# Ultrasonic Config
ULTRASONIC_TRIG = 19
ULTRASONIC_ECHO = 13
ULTRASONIC_THRESHOLD_M = 1.0
ULTRASONIC_MAX_DISTANCE_M = 4.0
GPIO_CHIP = 15
ULTRASONIC_MIN_VALID_M = 0.04

LED_PINS = {
    "far_left":  17,
    "mid_left":  23,
    "center":    22,
    "mid_right": 24,
    "far_right": 27,
}

state_lock = threading.Lock()
shared_state = {
    "lidar": {"far_left": False, "mid_left": False, "center": False, "mid_right": False, "far_right": False},
    "tof_1": {"mid_left": False, "center": False, "mid_right": False},
    # "tof_2": {...}  adding this when I figure out second ToF
    "ultrasonic_detected": False,
}

stop_event = threading.Event()

# LiDAR worker
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



# ToF worker (one sensor for now)
def tof_worker():
    driver = VL53L5CX()
    if not driver.is_alive():
        print("ToF sensor 1 not alive")
        return

    driver.init()
    driver.set_resolution(VL53L5CX_RESOLUTION_8X8)
    driver.set_ranging_frequency_hz(15)
    driver.start_ranging()

    def is_valid(status, distance):
        return status != 255 and distance < TOF_MAX_VALID_MM

    while not stop_event.is_set():
        zones = {"mid_left": False, "center": False, "mid_right": False}

        if driver.check_data_ready():
            ranging_data = driver.get_ranging_data()

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



# Ultrasonic worker
def ultrasonic_worker():
    h = lgpio.gpiochip_open(GPIO_CHIP)
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

            # 2 consecutive detection indicates obstacle
            hits = hits + 1 if close else 0
            with state_lock:
                shared_state["ultrasonic_detected"] = hits >= 2
    finally:
        lgpio.gpiochip_close(h)



# Fusion loop (driver of motors when in angular zones)
def fusion_loop():
    h = lgpio.gpiochip_open(GPIO_CHIP)
    for pin in LED_PINS.values():
        lgpio.gpio_claim_output(h, pin, 0)

    def set_led(zone, on):
        lgpio.gpio_write(h, LED_PINS[zone], 1 if on else 0)

    try:
        while True:
            with state_lock:
                lidar = shared_state["lidar"].copy()
                tof_1 = shared_state["tof_1"].copy()
                ultrasonic = shared_state["ultrasonic_detected"]

            far_left  = lidar["far_left"]
            far_right = lidar["far_right"]
            mid_left  = lidar["mid_left"] or tof_1["mid_left"]
            mid_right = lidar["mid_right"] or tof_1["mid_right"]

            center = lidar["center"] or tof_1["center"]
            if not lidar["center"] and ultrasonic:
                center = True

            set_led("far_left", far_left)
            set_led("mid_left", mid_left)
            set_led("center", center)
            set_led("mid_right", mid_right)
            set_led("far_right", far_right)

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for zone in LED_PINS:
            set_led(zone, False)
        lgpio.gpiochip_close(h)


if __name__ == "__main__":
    threads = [
        threading.Thread(target=lidar_worker, daemon=True),
        threading.Thread(target=tof_worker, daemon=True),
        threading.Thread(target=ultrasonic_worker, daemon=True),
    ]
    for t in threads:
        t.start()

    fusion_loop()
