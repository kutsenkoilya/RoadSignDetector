# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 20:10:45 2017
С этими не работает:
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\solidWhiteRight.mp4
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\solidYellowLeft.mp4
С этими работает
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\project_video.mp4
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\challenge_video.mp4
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\challenge.mp4

use this script to prepare data for target resolution
02_camClaibPickle.py
"""

from lib_curvedLaneDetector import detectAndDraw_CurvedLane
import cv2
import numpy as np
import sys

filepath = sys.argv[1]
cap = cv2.VideoCapture(filepath)
pauseMode = False

global window_search 
global frame_count
window_search = True
frame_count = 0
cache = None

while cap.isOpened():
    if (not pauseMode):
        ret, frame = cap.read()
        if (not ret):
            break
        
        frame = cv2.resize(frame, (640, 480)) 
        
        prcframe = detectAndDraw_CurvedLane(frame, window_search,frame_count,cache)
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