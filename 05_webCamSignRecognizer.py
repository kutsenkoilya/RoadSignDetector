# -*- coding: utf-8 -*-
"""
Распознаввтель квадратных знаков
Под данным с камеры

"""
from lib_squareSignDetector import findSquareTrafficSign
import cv2
import numpy as np
import sys

from imutils.perspective import four_point_transform
from imutils import contours
import imutils

#filepath = sys.argv[1]
camera = cv2.VideoCapture(0) #filepath
pauseMode = False

while camera.isOpened():
    if (not pauseMode):
        ret, frame = camera.read()
        if (not ret):
            break
        
        mod = findSquareTrafficSign(frame,False)
        
        cv2.imshow('modified', mod)
        cv2.imshow('original', frame)
    
    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break;
    if k ==32:
        pauseMode = not pauseMode    

camera.release()
cv2.destroyAllWindows()