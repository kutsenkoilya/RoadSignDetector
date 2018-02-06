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

def GoForward():
    ser.write(struct.pack('>3B', 125, 128, 2)) 
	
def GoBackward():
    ser.write(struct.pack('>3B', 125, 128, 0)) 

def TrunLeft():
    ser.write(struct.pack('>3B', 160, 128, 2)) 
        
def TurnRight():
    ser.write(struct.pack('>3B', 90, 128, 2)) 

def Stop():
    ser.write(struct.pack('>3B', 125, 128, 1)) 
    
def stopAll():
    ser.close()

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


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    
    cv2.imshow("Frame", image)
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
            if (key == ord('w')):
                GoForward()
                print('Going forward')
            if (key == ord('a')):
                TrunLeft()
                print('Going left')
            if (key == ord('s')):
                GoBackward()
                print('Going back')
            if (key == ord('d')):
                TurnRight()
                print('Going right')
            if (key == ord('f')):
                Stop()
                print('stop')
        key_p = key
    
    rawCapture.truncate(0)



