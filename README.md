# Self-Driving Car Project

## Overview
This repository contains the code for a self-driving car project utilizing both a Raspberry Pi and an Arduino. The Raspberry Pi handles image processing and decision-making, while the Arduino controls the motors based on the processed data. This project leverages sensors and machine learning techniques to enhance pathfinding and ensure safety, making it a hands-on exploration of autonomous driving technology.

## Repository Structure
- **Main Code For Raspberry.cpp**: Handles computer vision and high-level decision-making.
- **Main Code for Arduino.ino**: Controls motor movements based on Raspberry Pi commands.

## Installation, Usage & Additional Information
```sh
# Clone the Repository
git clone https://github.com/yourusername/self-driving-car.git
cd self-driving-car

# Upload the Arduino code
# 1. Open `Main Code for Arduino.ino` in Arduino IDE.
# 2. Select the correct board and port.
# 3. Compile and upload the code.

# Run the Raspberry Pi code
# Ensure OpenCV and required libraries are installed.
g++ -o car "Main Code For Raspberry.cpp" $(pkg-config --cflags --libs opencv4)
./car
```

### Requirements
- Raspberry Pi (any model with camera support)
- Arduino (Uno/Nano recommended)
- OpenCV (for image processing)
- Motor driver module
- Camera module
- Power supply

### Contribution
Feel free to fork the repository and submit pull requests. Ensure that code is well-documented and tested before submission.

### Contact
For any questions or collaborations, feel free to reach out via [sourbhr12@gmail.com] or open an issue in the repository.

### More About the Project
- [Google Drive Link](https://drive.google.com/drive/folders/13ktf_n2etD7QfB6YRN4_43YagsF_kYzc?usp=sharing)
