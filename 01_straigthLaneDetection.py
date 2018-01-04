# -*- coding: utf-8 -*-
"""
https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0 - распознавание полос движения. Только прямо

D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\solidWhiteRight.mp4
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\solidYellowLeft.mp4
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\project_video.mp4
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\challenge_video.mp4
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\challenge.mp4

"""

from lib_straightLaneDetector import detectAndDraw_StraightLane
import cv2
import numpy as np
import sys

filepath = sys.argv[1]
 
cap = cv2.VideoCapture(filepath)

first_frame = 1
cache = None

pauseMode = False

while cap.isOpened():
    if (not pauseMode):
        ret, frame = cap.read()
        if (not ret):
            break
        
        prcframe = detectAndDraw_StraightLane(frame,first_frame,cache)
        cache = frame
        
        cv2.imshow('original', frame)
        cv2.imshow('processed', prcframe)
    
    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break;
    if k ==32:
        pauseMode = not pauseMode    

cap.release()
cv2.destroyAllWindows()