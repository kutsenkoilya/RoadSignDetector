# -*- coding: utf-8 -*-
"""
https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0 - распознавание полос движения. Только прямо

"""

import cv2
import numpy as np
import sys

cap = cv2.VideoCapture(0)


pauseMode = False

def Sensor_LaneCalculator(sensor,maxWidth):
    edgeArr = np.where( sensor == 255 )
    if (edgeArr is not None and edgeArr[1] is not None and len(edgeArr[1])>0):
        minInd = edgeArr[1][0]
        maxInd = edgeArr[1][-1]
        if (maxInd - minInd <= maxWidth):
            return int((maxInd+minInd)/2)
    return 0
               

while cap.isOpened():
    if (not pauseMode):
        ret, frame = cap.read()
        if (not ret):
            break
        
        height, width, _ = frame.shape
        
        croppedImg = frame[int(height/2):height,int(width/2):width]
        cannyImg = cv2.Canny(croppedImg,100,200)
        
        heightR, widthR = cannyImg.shape
        maxWidth = widthR/10
        sensor_step = 40
        
        pointsArr = []
        
        for i in range(0,heightR,sensor_step):
            sensor = cannyImg[i:i+1,:]
            avg = Sensor_LaneCalculator(sensor,maxWidth)
            if (avg != 0):
                point = tuple([avg,i])
                pointsArr.append(point)
                
        
        if (len(pointsArr)>2):
            cv2.line(croppedImg, pointsArr[0],pointsArr[-1],(0,0,0), 2)
            
            for point in pointsArr:
                cv2.circle(croppedImg, point,4,(0,255,255),4)
            
            
        cv2.imshow('original', frame)        
        
    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break;
    if k ==32:
        pauseMode = not pauseMode    

cap.release()
cv2.destroyAllWindows()