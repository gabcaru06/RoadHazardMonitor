# Wearable Assistive Technology Solution for Enhancing Mobility in Visually Impaired Individuals

# User Help
**This code requires the following:**
- Latest version of [Python](https://www.python.org/downloads/)

## Obstacle Detection Dependencies

### Arduino Testing
- `VL53L0X` by Pololu - install via Arduino IDE Library Manager
  - Repo: https://github.com/pololu/vl53l0x-arduino

### Python Data Testing
- `pyserial` - install via
  ```bash
  pip install pyserial
  ```
  
### Raspberry Pi Setup
- Tools
  ```bash
  sudo apt update
  sudo apt install -y git cmake pkg-config build-essential swig python3-dev python3-pip python3-smbus i2c-tools
  ```
  
#### YDLiDAR Setup
- YDLiDAR Repository: https://github.com/YDLIDAR/YDLidar-SDK
- YDLiDAR Build and Install SDK:
  ```bash
  git clone https://github.com/YDLIDAR/YDLidar-SDK.git
  cd YDLidar-SDK
  mkdir build && cd build
  cmake ..
  make
  sudo make install
  ```
  
#### ToF Setup
- ToF Repository:
- ToF Build and Install:
  ```bash
  git clone <INSERT VL53L5CX REPO URL>
  cd <repo-folder>
  pip install .
  ```
- Confirm Sensor is detected:
  ```bash
  i2cdetect -y 1
  ```
  

## Computer Vision Dependencies
