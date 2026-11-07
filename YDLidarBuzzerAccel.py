import ydlidar
import time
import struct
import threading
from gpiozero import DigitalOutputDevice
from smbus2 import SMBus  # pip install smbus2  (and enable I2C: sudo raspi-config)

# Lidar Config
PORT = "/dev/ttyUSB0"
BAUDRATE = 921600
THRESHOLD_M = 1.0

# Buzzer pulse config (all of this is similar to implementation in arduino script)
CLOSE_M = 0.05        # Max close distance (uses max frequency)
FAR_M = THRESHOLD_M   # Above threshold (no buzz)
BASE_FREQ_HZ = 1.0    # slow pulse, just detected
MAX_FREQ_HZ = 8.0     # fast pulse, very close
DUTY_CYCLE = 0.3

# Buzzer GPIO pin setup
BUZZER_PINS = [17, 23, 22, 24, 27]
NUM_ZONES = len(BUZZER_PINS)

# Motion / idle config (mirrors proximity_haptic_idle.ino)
# Wiring (Raspberry Pi): GY-521 VCC -> 3.3V (or 5V), GND -> GND,
#                        SDA -> GPIO2 (pin 3), SCL -> GPIO3 (pin 5)
I2C_BUS = 1
MPU_ADDR = 0x68
IDLE_TIMEOUT_S = 5.0        # still this long -> idle
ACCEL_THRESH_G = 0.15       # change between samples (tune)
GYRO_THRESH_DPS = 4.0       # deg/s above resting bias (tune)
MOTOR_BLANKING_S = 0.06     # ignore MPU just after a buzzer was on
LIDAR_SETTLE_S = 1.5        # ignore MPU while LiDAR motor spins up/down
MOTION_INTERVAL_S = 0.05
DEBUG_MOTION = True         # print live motion numbers (lines starting "# motion")


# <-40  -40..-20  -20..20  20..40  >40
def get_zone(angle_deg):
    if angle_deg < -40:
        return 0
    elif -40 <= angle_deg < -20:
        return 1
    elif -20 <= angle_deg <= 20:
        return 2
    elif 20 < angle_deg <= 40:
        return 3
    else:  # > 40
        return 4


def distance_to_freq(distance_m):
    """Closer obstacle -> faster pulse. None / out of range -> 0 (silent)."""
    if distance_m is None or distance_m >= FAR_M:
        return 0.0
    if distance_m <= CLOSE_M:
        urgency = 1.0
    else:
        urgency = (FAR_M - distance_m) / (FAR_M - CLOSE_M)
    return BASE_FREQ_HZ + urgency * (MAX_FREQ_HZ - BASE_FREQ_HZ)


class HapticBuzzers:
    """
    Drives one buzzer/motor per zone, pulsing each at its own frequency.

    Pulsing runs in a background thread so it keeps its timing independent of
    the LiDAR scan rate (the scan loop only updates the target frequencies),
    like updateBuzzer() running independently of the sensor read in the .ino.
    """

    def __init__(self, pins):
        self._devices = [DigitalOutputDevice(p) for p in pins]
        self._freqs = [0.0] * len(pins)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        # Last time any buzzer was driven on; the motion monitor uses this to
        # ignore vibration caused by the buzzers themselves.
        self.last_active = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_frequencies(self, freqs):
        with self._lock:
            self._freqs = list(freqs)

    def _run(self):
        while not self._stop_event.is_set():
            now = time.monotonic()
            with self._lock:
                freqs = list(self._freqs)
            any_on = False
            for dev, f in zip(self._devices, freqs):
                if f <= 0.0:
                    dev.off()
                    continue
                period = 1.0 / f
                if (now % period) < period * DUTY_CYCLE:
                    dev.on()
                    any_on = True
                else:
                    dev.off()
            if any_on:
                self.last_active = now
            self._stop_event.wait(0.005)

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        for dev in self._devices:
            dev.off()
            dev.close()


class MotionMonitor:
    """
    Polls an MPU-6050 in a background thread and sets `idle` to True after
    IDLE_TIMEOUT_S without motion; any motion clears it again.

    Same approach as the .ino: motion = change in accel between samples, or
    gyro rate above its resting bias. Samples are skipped while the buzzers
    are vibrating (or the LiDAR motor is changing speed), and a fresh
    baseline is taken afterwards so that vibration isn't mistaken for motion.
    """

    def __init__(self, buzzers):
        self._buzzers = buzzers
        self._bus = SMBus(I2C_BUS)
        self._bus.write_byte_data(MPU_ADDR, 0x6B, 0)  # wake MPU-6050 (starts asleep)
        time.sleep(0.1)

        self._gyro_bias = self._calibrate_gyro()
        first = self._read()
        self._prev_accel = first[0] if first else (0.0, 0.0, 0.0)

        self._need_baseline = True
        self._blank_until = 0.0
        self._last_motion = time.monotonic()
        self._idle = False
        self._d_accel = 0.0
        self._gyro_mag = 0.0

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def idle(self):
        return self._idle

    def notify_disturbance(self):
        """Call after the LiDAR motor starts/stops: ignore the MPU briefly."""
        self._blank_until = time.monotonic() + LIDAR_SETTLE_S
        self._need_baseline = True

    def _read(self):
        try:
            data = self._bus.read_i2c_block_data(MPU_ADDR, 0x3B, 14)
        except OSError:
            return None
        # ax, ay, az, temp, gx, gy, gz  (big-endian signed 16-bit)
        ax, ay, az, _, gx, gy, gz = struct.unpack(">hhhhhhh", bytes(data))
        accel = (ax / 16384.0, ay / 16384.0, az / 16384.0)   # g (+/-2g)
        gyro = (gx / 131.0, gy / 131.0, gz / 131.0)          # deg/s (+/-250)
        return accel, gyro

    def _calibrate_gyro(self):
        """Keep the device still at startup."""
        sums = [0.0, 0.0, 0.0]
        n = 0
        for _ in range(40):
            r = self._read()
            if r:
                for i in range(3):
                    sums[i] += r[1][i]
                n += 1
            time.sleep(0.025)
        return tuple(s / n for s in sums) if n else (0.0, 0.0, 0.0)

    def _motion_detected(self):
        r = self._read()
        if r is None:
            return False
        accel, gyro = r
        self._d_accel = sum(abs(a - p) for a, p in zip(accel, self._prev_accel))
        self._prev_accel = accel
        self._gyro_mag = sum(abs(g - b) for g, b in zip(gyro, self._gyro_bias))
        return self._d_accel > ACCEL_THRESH_G or self._gyro_mag > GYRO_THRESH_DPS

    def _run(self):
        last_debug = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            quiet = (now - self._buzzers.last_active > MOTOR_BLANKING_S
                     and now >= self._blank_until)

            if not quiet:
                self._need_baseline = True  # readings now are polluted
            elif self._need_baseline:
                # First quiet sample: take a clean baseline only
                r = self._read()
                if r:
                    self._prev_accel = r[0]
                self._need_baseline = False
            elif self._motion_detected():
                self._last_motion = now
                self._idle = False

            if DEBUG_MOTION and now - last_debug >= 0.2:
                last_debug = now
                print(f"# motion dAccel={self._d_accel:.3f} (thr {ACCEL_THRESH_G:.2f}) "
                      f"gyro={self._gyro_mag:.2f} (thr {GYRO_THRESH_DPS:.1f}) "
                      f"quiet={int(quiet)} stillFor={now - self._last_motion:.1f}s")

            if not self._idle and now - self._last_motion >= IDLE_TIMEOUT_S:
                self._idle = True

            self._stop_event.wait(MOTION_INTERVAL_S)

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._bus.close()


def main():
    ydlidar.os_init()

    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, PORT)
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, BAUDRATE)
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

    ret = laser.initialize()
    if not ret:
        print("Failed to initialize LiDAR:", laser.DescribeError())
        return

    ret = laser.turnOn()
    if not ret:
        print("Failed to turn on LiDAR:", laser.DescribeError())
        return

    buzzers = HapticBuzzers(BUZZER_PINS)
    try:
        monitor = MotionMonitor(buzzers)
    except OSError as e:
        print("MPU-6050 not found / I2C error:", e)
        buzzers.close()
        laser.turnOff()
        laser.disconnecting()
        return

    scan = ydlidar.LaserScan()
    lidar_on = True

    try:
        while ydlidar.os_isOk():
            # ---- Idle: no motion -> silence buzzers and stop the LiDAR ----
            if monitor.idle:
                if lidar_on:
                    buzzers.set_frequencies([0.0] * NUM_ZONES)
                    laser.turnOff()
                    lidar_on = False
                    monitor.notify_disturbance()  # motor spin-down isn't "motion"
                    print("# IDLE: no motion, LiDAR + buzzers OFF")
                time.sleep(0.1)
                continue

            # ---- Woke up: restart the LiDAR ----
            if not lidar_on:
                if laser.turnOn():
                    lidar_on = True
                    monitor.notify_disturbance()  # motor spin-up isn't "motion"
                    print("# WAKE: motion detected, LiDAR ON")
                else:
                    print("Failed to turn on LiDAR:", laser.DescribeError())
                    time.sleep(1.0)
                    continue

            result = laser.doProcessSimple(scan)

            # Closest obstacle distance per zone for this scan (None = clear)
            closest = [None] * NUM_ZONES

            if result:
                for point in scan.points:
                    distance_m = point.range
                    angle_deg = point.angle * 180.0 / 3.14159265

                    if 0 < distance_m <= THRESHOLD_M:
                        zone = get_zone(angle_deg)
                        if closest[zone] is None or distance_m < closest[zone]:
                            closest[zone] = distance_m

                freqs = [distance_to_freq(d) for d in closest]
                buzzers.set_frequencies(freqs)

                for zone, d in enumerate(closest):
                    if d is not None:
                        print(f"ALERT: zone {zone}, closest {d:.3f} m, "
                              f"buzz {freqs[zone]:.1f} Hz")
            else:
                print("Failed to get scan data.")
                # No data -> silence rather than keep buzzing on stale readings
                buzzers.set_frequencies([0.0] * NUM_ZONES)

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        monitor.close()
        buzzers.close()
        laser.turnOff()
        laser.disconnecting()


if __name__ == "__main__":
    main()
