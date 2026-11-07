import ydlidar
import time
import threading
from gpiozero import DigitalOutputDevice

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
            for dev, f in zip(self._devices, freqs):
                if f <= 0.0:
                    dev.off()
                    continue
                period = 1.0 / f
                if (now % period) < period * DUTY_CYCLE:
                    dev.on()
                else:
                    dev.off()
            self._stop_event.wait(0.005)

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        for dev in self._devices:
            dev.off()
            dev.close()


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
    scan = ydlidar.LaserScan()

    try:
        while ydlidar.os_isOk():
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
        buzzers.close()
        laser.turnOff()
        laser.disconnecting()


if __name__ == "__main__":
    main()
