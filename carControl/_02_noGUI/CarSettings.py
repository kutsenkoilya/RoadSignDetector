# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 01:28:05 2018

@author: Илья
"""

#CAMERA SETTINGS
PiCameraResH = 240
PiCameraResW = 320
PiCameraFrameRate = 32
RecEnabled = 0

#MOVEMENT SETTINGS
StartSpeed = 125


#RightLane SETTINGS
CroppedH = int(PiCameraResH/2)
CroppedW = int(PiCameraResW/2)
LaneMaxW = int(CroppedW/10)
SensorStep = 40

