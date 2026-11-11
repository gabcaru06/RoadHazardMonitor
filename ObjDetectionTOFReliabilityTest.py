import time
import csv
from vl53l5cx.vl53l5cx import VL53L5CX, VL53L5CX_RESOLUTION_8X8

# ---- CONFIG ----
TOF_MAX_VALID_MM = 4000
TOF_CENTER_COLS = (2, 3, 4, 5)
HIT_TOLERANCE_MM = 5

TEST_DURATION_MIN = 30
SAMPLE_INTERVAL_S = 0.1  # ToF runs much faster than LiDAR/US, so a tighter interval makes sense here


def is_valid(status, distance):
    return status != 255 and distance < TOF_MAX_VALID_MM


def is_hit(distance_mm, known_distance_mm, tolerance_mm):
    if distance_mm is None:
        return False
    return abs(distance_mm - known_distance_mm) <= tolerance_mm


if __name__ == "__main__":
    known_distance_m = float(input("Enter the known distance you're placing the obstacle at (in meters): "))
    known_distance_mm = known_distance_m * 1000
    print(f"Using known distance: {known_distance_mm:.1f} mm, hit tolerance: ±{HIT_TOLERANCE_MM} mm")

    print("\nReposition the ToF sensor to point straight ahead (not its usual downward angle) for this test.")
    input(f"Place the object at exactly {known_distance_m} m from the sensor, then press Enter to start...")

    driver = VL53L5CX()
    if not driver.is_alive():
        raise IOError("VL53L5CX Device is not alive")

    driver.init()
    driver.set_resolution(VL53L5CX_RESOLUTION_8X8)
    driver.set_ranging_frequency_hz(15)
    driver.start_ranging()

    total_samples = 0
    hits = 0
    current_miss_streak = 0
    longest_miss_streak = 0

    test_duration_s = TEST_DURATION_MIN * 60
    start_time = time.time()

    print(f"Logging for {TEST_DURATION_MIN} minutes. Do not disturb the object or sensor.")

    try:
        with open("tof_reliability_test.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "known_distance_mm", "tof_mm", "tof_hit"])

            while time.time() - start_time < test_duration_s:
                if driver.check_data_ready():
                    ranging_data = driver.get_ranging_data()
                    valid = []
                    for i in range(64):
                        status = ranging_data.target_status[driver.nb_target_per_zone * i]
                        distance = ranging_data.distance_mm[driver.nb_target_per_zone * i]
                        col = i % 8
                        if col in TOF_CENTER_COLS and is_valid(status, distance):
                            valid.append(distance)

                    tof_mm = sum(valid) / len(valid) if valid else None
                    elapsed_s = round(time.time() - start_time, 2)
                    hit = is_hit(tof_mm, known_distance_mm, HIT_TOLERANCE_MM)

                    writer.writerow([elapsed_s, known_distance_mm, tof_mm, hit])

                    total_samples += 1
                    if hit:
                        hits += 1
                        current_miss_streak = 0
                    else:
                        current_miss_streak += 1
                        longest_miss_streak = max(longest_miss_streak, current_miss_streak)

                time.sleep(SAMPLE_INTERVAL_S)

        print("\n=== ToF Reliability Test Results ===")
        print(f"Known distance: {known_distance_mm:.1f} mm (±{HIT_TOLERANCE_MM} mm tolerance)")
        print(f"Total samples: {total_samples}")
        hit_rate = (hits / total_samples * 100) if total_samples else 0
        longest_miss_s = longest_miss_streak * SAMPLE_INTERVAL_S
        print(f"Hit rate: {hit_rate:5.1f}%  |  longest miss streak: {longest_miss_streak} samples ({longest_miss_s:.1f}s)")

        print("\nSaved to tof_reliability_test.csv")

    finally:
        driver.stop_ranging()
