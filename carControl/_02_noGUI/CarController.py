# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 00:23:09 2018

@author: Илья
"""

import CarSettings as CarSettings

import struct
import serial

import time

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np

def Forward(spd):
    ser.write(struct.pack('>3B', 125, spd, 2)) 
	
def ForwardLeft(spd):
    ser.write(struct.pack('>3B', 160, spd, 2)) 
        
def ForwardRight(spd):
    ser.write(struct.pack('>3B', 90, spd, 2)) 

def Backward(spd):
    ser.write(struct.pack('>3B', 125, spd, 0)) 

def BackwardLeft(spd):
    ser.write(struct.pack('>3B', 160, spd, 0)) 

def BackwardRight(spd):
    ser.write(struct.pack('>3B', 90, spd, 0)) 

def Stop():
    ser.write(struct.pack('>3B', 125, 0, 1)) 
    
def stopAll():
    ser.close()

def Sensor_LaneCalculator(sensor):
    edgeArr = np.where( sensor == 255 )
    if (edgeArr is not None and edgeArr[1] is not None and len(edgeArr[1])>0):
        minInd = edgeArr[1][0]
        maxInd = edgeArr[1][-1]
        if (maxInd - minInd <= CarSettings.LaneMaxW):
            return int((maxInd+minInd)/2)
    return 0

first_image = 1
cache = None

recEnabled = CarSettings.RecEnabled

camera = PiCamera()
camera.resolution = (CarSettings.PiCameraResW,CarSettings.PiCameraResH)
camera.framerate = CarSettings.PiCameraFrameRate
camera.vflip = True
rawCapture = PiRGBArray(camera, size=(CarSettings.PiCameraResW,CarSettings.PiCameraResH))

time.sleep(0.1)

if (recEnabled == 1):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (CarSettings.PiCameraResW,CarSettings.PiCameraResH))

ser = serial.Serial('/dev/ttyUSB0', 9600)

key_p = None
wait_time = 1

speed = CarSettings.StartSpeed
print("car speed: {}".format(speed))

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    image = frame.array
    
    croppedImg = image[CarSettings.CroppedH:CarSettings.PiCameraResH,
                       CarSettings.CroppedW:CarSettings.PiCameraResW]
    
    cannyImg = cv2.Canny(croppedImg,100,200)
    
    pointsArr = []
    
    for i in range(0,CarSettings.CroppedH,CarSettings.SensorStep):
        sensor = cannyImg[i:i+1,:]
        avg = Sensor_LaneCalculator(sensor)
        if (avg != 0):
            point = tuple([avg,i])
            pointsArr.append(point)
    
    if (len(pointsArr)>2):
        cv2.line(croppedImg, pointsArr[0],pointsArr[-1],(0,0,0), 2)
        for point in pointsArr:
            cv2.circle(croppedImg, point,4,(0,255,255),4)
    
    cv2.imshow("Frame", image)
    cv2.imshow("cropped", croppedImg)
    
    if (recEnabled == 1):
        out.write(image)

    key = cv2.waitKey(1)
    
    if (key != -1):
        if (key != key_p):
            if (key == ord('x')):
                print('exiting')
                cv2.destroyAllWindows()
                stopAll()
                if (recEnabled == 1):
                    out.release()
                break
            
            if (key == ord('q')):
                ForwardLeft(speed)
                print('Fwd Left')
            if (key == ord('w')):
                Forward(speed)
                print('Fwd')
            if (key == ord('e')):
                ForwardRight(speed)
                print('Fwd Right')

            if (key == ord('a')):
                BackwardLeft(speed)
                print('Bck Left')
            if (key == ord('s')):
                Backward(speed)
                print('Bck')
            if (key == ord('d')):
                BackwardRight(speed)
                print('Bck Right')

            if (key == ord('f')):
                Stop()
                print('stop')

            if (key == ord('[')):
                speed = speed - 50
                if (speed < 50):
                    speed = 50 
                print("car speed: {}".format(speed))   
            if (key == ord(']')):
                speed = speed + 50
                if (speed > 255):
                    speed = 255
                print("car speed: {}".format(speed))

        key_p = key
    
    rawCapture.truncate(0)



