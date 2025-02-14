# this code is only for raspberryPi as windows do not support some libraries whiich are mentioned below

import cv2
import numpy as np
import time
import RPi.GPIO as GPIO

# Image Processing variables
frame, framePers, frameGray, frameThresh, frameEdge, frameFinal = [None] * 6
frameFinalDuplicate, frameFinalDuplicate1, ROILane, ROILaneEnd = [None] * 4
LeftLanePos, RightLanePos, frameCenter, laneCenter, Result, laneEnd = [0] * 6
histogramLane, histogramLaneEnd = [], []

# Define Source and Destination points for perspective transform
Source = np.array([[40, 135], [360, 135], [0, 185], [400, 185]], dtype='float32')
Destination = np.array([[100, 0], [280, 0], [100, 240], [280, 240]], dtype='float32')

# Set up GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(23, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)

# Initialize the Cascade Classifiers
Stop_Cascade = cv2.CascadeClassifier('/home/pi/Desktop/MACHINE_LEARNING/Stop_cascade.xml')
Object_Cascade = cv2.CascadeClassifier('/home//pi/Desktop/MACHINE_LEARNING/Object_cascade.xml')

# Camera Setup
def Setup():
    global Camera
    Camera = cv2.VideoCapture(0)
    if not Camera.isOpened():
        print("Error: Could not open the camera.")
        exit(1)
    
    settings = {
        cv2.CAP_PROP_FRAME_WIDTH: 400,
        cv2.CAP_PROP_FRAME_HEIGHT: 240,
        cv2.CAP_PROP_BRIGHTNESS: 50,
        cv2.CAP_PROP_CONTRAST: 50,
        cv2.CAP_PROP_SATURATION: 50,
        cv2.CAP_PROP_GAIN: 50,
        cv2.CAP_PROP_FPS: 30
    }
    for prop, value in settings.items():
        Camera.set(prop, value)

def Capture():
    global frame
    ret, frame = Camera.read()
    if not ret:
        print("Failed to capture frame.")
        exit(1)
    frame_Stop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_Object = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def Perspective():
    global framePers
    frame = cv2.polylines(frame, [Source.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)
    Matrix = cv2.getPerspectiveTransform(Source, Destination)
    framePers = cv2.warpPerspective(frame, Matrix, (400, 240))

def Threshold():
    global frameFinal, frameFinalDuplicate, frameFinalDuplicate1
    frameGray = cv2.cvtColor(framePers, cv2.COLOR_RGB2GRAY)
    _, frameThresh = cv2.threshold(frameGray, 230, 255, cv2.THRESH_BINARY)
    frameEdge = cv2.Canny(frameGray, 900, 900, apertureSize=3, L2gradient=False)
    frameFinal = cv2.add(frameThresh, frameEdge)
    frameFinal = cv2.cvtColor(frameFinal, cv2.COLOR_GRAY2RGB)
    frameFinalDuplicate = cv2.cvtColor(frameFinal, cv2.COLOR_RGB2BGR)  # used in histogram function
    frameFinalDuplicate1 = cv2.cvtColor(frameFinal, cv2.COLOR_RGB2BGR)  # used in histogram function

def Histogram():
    global laneEnd
    histogramLane.clear()
    for i in range(400):  # frame.size().width = 400
        ROILane = frameFinalDuplicate[i:140, i:141]
        ROILane = 255 / ROILane
        histogramLane.append(int(np.sum(ROILane)))
    
    histogramLaneEnd.clear()
    for i in range(400):
        ROILaneEnd = frameFinalDuplicate1[i:240, i:241]
        ROILaneEnd = 255 / ROILaneEnd
        histogramLaneEnd.append(int(np.sum(ROILaneEnd)))
    
    laneEnd = np.sum(histogramLaneEnd)
    print("Lane END =", laneEnd)

def LaneFinder():
    global LeftLanePos, RightLanePos
    LeftLanePos = np.argmax(histogramLane[:150])
    RightLanePos = np.argmax(histogramLane[250:])
    
    cv2.line(frameFinal, (LeftLanePos, 0), (LeftLanePos, 240), (0, 255, 0), 2)
    cv2.line(frameFinal, (RightLanePos, 0), (RightLanePos, 240), (0, 255, 0), 2)

def LaneCenter():
    global laneCenter, frameCenter, Result
    laneCenter = (RightLanePos - LeftLanePos) // 2 + LeftLanePos
    frameCenter = 188
    cv2.line(frameFinal, (laneCenter, 0), (laneCenter, 240), (0, 255, 0), 3)
    cv2.line(frameFinal, (frameCenter, 0), (frameCenter, 240), (255, 0, 0), 3)
    Result = laneCenter - frameCenter

def Stop_detection():
    global dist_Stop
    if not Stop_Cascade.empty():
        RoI_Stop = frame_Stop[200:340, 0:140]
        gray_Stop = cv2.cvtColor(RoI_Stop, cv2.COLOR_RGB2GRAY)
        gray_Stop = cv2.equalizeHist(gray_Stop)
        Stop = Stop_Cascade.detectMultiScale(gray_Stop)
        
        for (x, y, w, h) in Stop:
            cv2.rectangle(RoI_Stop, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(RoI_Stop, "Stop Sign", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            dist_Stop = (-1.07) * (w) + 102.597
            cv2.putText(RoI_Stop, f"D = {dist_Stop} cm", (1, 130), 0, 1, (0, 0, 255), 2)

def Object_detection():
    global dist_Object
    if not Object_Cascade.empty():
        RoI_Object = frame_Object[100:290, 50:240]
        gray_Object = cv2.cvtColor(RoI_Object, cv2.COLOR_RGB2GRAY)
        gray_Object = cv2.equalizeHist(gray_Object)
        Object = Object_Cascade.detectMultiScale(gray_Object)
        
        for (x, y, w, h) in Object:
            cv2.rectangle(RoI_Object, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(RoI_Object, "Object", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            dist_Object = (-0.48) * (w) + 56.6
            cv2.putText(RoI_Object, f"D = {dist_Object} cm", (1, 130), 0, 1, (0, 0, 255), 2)

def main():
    Setup()
    print("Connecting to camera")
    
    if not Camera.isOpened():
        print("Failed to connect")
        return
    
    while True:
        start_time = time.time()
        
        Capture()
        Perspective()
        Threshold()
        Histogram()
        LaneFinder()
        LaneCenter()
        Stop_detection()
        Object_detection()
        
        if 5 < dist_Stop < 20:
            GPIO.output(21, GPIO.LOW)
            GPIO.output(22, GPIO.LOW)
            GPIO.output(23, GPIO.LOW)
            GPIO.output(24, GPIO.HIGH)
            print("Stop Sign Detected")
            dist_Stop = 0
            continue
        
        if 5 < dist_Object < 20:
            GPIO.output(21, GPIO.HIGH)
            GPIO.output(22, GPIO.LOW)
            GPIO.output(23, GPIO.LOW)
            GPIO.output(24, GPIO.HIGH)
            print("Object Detected")
            dist_Object = 0
            continue
        
        if laneEnd > 3000:
            GPIO.output(21, GPIO.HIGH)
            GPIO.output(22, GPIO.HIGH)
            GPIO.output(23, GPIO.HIGH)
            GPIO.output(24, GPIO.LOW)
            print("Lane End Detected")
        
        if Result == 0:
            GPIO.output(21, GPIO.LOW)
            GPIO.output(22, GPIO.LOW)
            GPIO.output(23, GPIO.LOW)
            GPIO.output(24, GPIO.LOW)
            print("Move Forward")
            
    # Right turn logic
    if 0 < Result < 10:
        GPIO.output(21, GPIO.HIGH)
        GPIO.output(22, GPIO.LOW)
        GPIO.output(23, GPIO.LOW)
        GPIO.output(24, GPIO.LOW)
        print("Right1")
    elif 10 <= Result < 20:
        GPIO.output(21, GPIO.LOW)
        GPIO.output(22, GPIO.HIGH)
        GPIO.output(23, GPIO.LOW)
        GPIO.output(24, GPIO.LOW)
        print("Right2")
    elif Result >= 20:
        GPIO.output(21, GPIO.HIGH)
        GPIO.output(22, GPIO.HIGH)
        GPIO.output(23, GPIO.LOW)
        GPIO.output(24, GPIO.LOW)
        print("Right3")

    # Left turn logic
    elif -10 < Result < 0:
        GPIO.output(21, GPIO.LOW)
        GPIO.output(22, GPIO.LOW)
        GPIO.output(23, GPIO.HIGH)
        GPIO.output(24, GPIO.LOW)
        print("Left1")
    elif -20 < Result <= -10:
        GPIO.output(21, GPIO.HIGH)
        GPIO.output(22, GPIO.LOW)
        GPIO.output(23, GPIO.HIGH)
        GPIO.output(24, GPIO.LOW)
        print("Left2")
    elif Result <= -20:
        GPIO.output(21, GPIO.LOW)
        GPIO.output(22, GPIO.HIGH)
        GPIO.output(23, GPIO.HIGH)
        GPIO.output(24, GPIO.LOW)
        print("Left3")

    # Lane End Detection
    if laneEnd > 3000:
        print("Lane End")
        # frame.putText("Lane End", (1, 50), font, scale, (255, 0, 0), 2)

    # Move Forward logic
    elif Result == 0:
        print(f"Result = {Result} (Move Forward)")
        # frame.putText(f"Result = {Result} (Move Forward)", (1, 50), font, scale, (0, 0, 255), 2)

    # Move Right logic
    elif Result > 0:
        print(f"Result = {Result} (Move Right)")
        # frame.putText(f"Result = {Result} (Move Right)", (1, 50), font, scale, (0, 0, 255), 2)

    # Move Left logic
    elif Result < 0:
        print(f"Result = {Result} (Move Left)")
        # frame.putText(f"Result = {Result} (Move Left)", (1, 50), font, scale, (0, 0, 255), 2)
        
        # Display images
        cv2.imshow("Original", frame)
        cv2.imshow("Perspective", framePers)
        cv2.imshow("Final", frameFinal)
        cv2.imshow("Stop Sign", RoI_Stop)
        cv2.imshow("Object", RoI_Object)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print(f"FPS: {1 / (time.time() - start_time):.2f}")
    
    Camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
