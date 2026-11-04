import ydlidar
import time
from gpiozero import LED

# Lidar Config
PORT = "/dev/ttyUSB0"
BAUDRATE = 921600
THRESHOLD_M = 1

# Led Setup
led_far_left   = LED(17)  # < -40
led_mid_left   = LED(23)  # -40 to -20
led_center     = LED(22)  # -20 to 20
led_mid_right  = LED(24)  # 20 to 40
led_far_right  = LED(27)  # > 40

leds = [led_far_left, led_mid_left, led_center, led_mid_right, led_far_right]

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

    scan = ydlidar.LaserScan()

    try:
        while ydlidar.os_isOk():
            result = laser.doProcessSimple(scan)

            # Track which zones have an obstacle this scan
            zone_triggered = [False, False, False, False, False]

            if result:
                for point in scan.points:
                    distance_m = point.range
                    angle_deg = point.angle * 180.0 / 3.14159265

                    if 0 < distance_m <= THRESHOLD_M:
                        zone = get_zone(angle_deg)
                        zone_triggered[zone] = True
                        print(f"ALERT: obstacle at {distance_m:.3f} m, angle {angle_deg:.2f} deg, zone {zone}")
            else:
                print("Failed to get scan data.")

            # Update LEDs to reflect current scan's results
            for i, led in enumerate(leds):
                if zone_triggered[i]:
                    led.on()
                else:
                    led.off()

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        for led in leds:
            led.off()
        laser.turnOff()
        laser.disconnecting()

if __name__ == "__main__":
    main()
