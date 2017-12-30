# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 18:49:17 2017

@author: Илья
"""
import cv2
import numpy as np

from imutils.perspective import four_point_transform
from imutils import contours
import imutils

def filterByColor_Blue(hsv, cv_debug):
    #hsv - image in hsv
    lower_blue = np.array([85,100,70])
    upper_blue = np.array([115,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    if (cv_debug):
        cv2.imshow('filtered color', mask)
    
    return mask

def findRoundTrafficSign(frame, cv_debug=False):
    
    frame = imutils.resize(frame, width=500)
    frameArea = frame.shape[0]*frame.shape[1]
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = filterByColor_Blue(hsv, cv_debug)
    
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    if (cv_debug):
        cv2.imshow('img w morphology', mask)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    if (cv_debug):
        print('number of countours:')
        print(len(cnts))
    
    boxes = []
    areas = []
    circles = []
    
    if len(cnts) > 0:
        for cnt in cnts:
            rect = cv2.minAreaRect(cnt)
            (cx,cy),rad = cv2.minEnclosingCircle(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            sideOne = np.linalg.norm(box[0]-box[1])
            sideTwo = np.linalg.norm(box[0]-box[3])
            area = sideOne*sideTwo
            
            boxes.append(box)
            areas.append(area)
            circles.append(((cx,cy),rad))
            
    for index, box in enumerate(boxes):
        area = areas[index]
        (cx,cy),rad = circles[index]
        
        if (2*rad < frame.shape[0] and 2*rad < frame.shape[1]):
        
            if (area > frameArea*0.02):
                cv2.drawContours(frame,[box],0,(0,0,255),2)
                cv2.circle(frame,(int(cx),int(cy)),int(rad),(0,255,0),2)
            
            if ([box][0] is not None):
                warped = four_point_transform(mask, [box][0])
                detectedTrafficSign = identifyTrafficSign(warped)
                cv2.putText(frame, detectedTrafficSign, tuple(box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            
    return frame
    
    

def identifyTrafficSign(image):
    SIGNS_LOOKUP = {
        (1, 0, 0, 1): 'Turn Right', # turnRight
        (0, 0, 1, 1): 'Turn Left', # turnLeft
        (0, 1, 0, 1): 'Move Forward', # moveStraight
        (1, 0, 1, 1): 'Turn Back', # turnBack
    }

    THRESHOLD = 150
    
    image = cv2.bitwise_not(image)
    (subHeight, subWidth) = np.divide(image.shape, 10)
    subHeight = int(subHeight)
    subWidth = int(subWidth)

    cv2.rectangle(image, (subWidth, 4*subHeight), (3*subWidth, 9*subHeight), (0,255,0),2) # left block
    cv2.rectangle(image, (4*subWidth, 4*subHeight), (6*subWidth, 9*subHeight), (0,255,0),2) # center block
    cv2.rectangle(image, (7*subWidth, 4*subHeight), (9*subWidth, 9*subHeight), (0,255,0),2) # right block
    cv2.rectangle(image, (3*subWidth, 2*subHeight), (7*subWidth, 4*subHeight), (0,255,0),2) # top block

    leftBlock = image[4*subHeight:9*subHeight, subWidth:3*subWidth]
    centerBlock = image[4*subHeight:9*subHeight, 4*subWidth:6*subWidth]
    rightBlock = image[4*subHeight:9*subHeight, 7*subWidth:9*subWidth]
    topBlock = image[2*subHeight:4*subHeight, 3*subWidth:7*subWidth]

    leftFraction = np.sum(leftBlock)/(leftBlock.shape[0]*leftBlock.shape[1])
    centerFraction = np.sum(centerBlock)/(centerBlock.shape[0]*centerBlock.shape[1])
    rightFraction = np.sum(rightBlock)/(rightBlock.shape[0]*rightBlock.shape[1])
    topFraction = np.sum(topBlock)/(topBlock.shape[0]*topBlock.shape[1])

    segments = (leftFraction, centerFraction, rightFraction, topFraction)
    segments = tuple(1 if segment > THRESHOLD else 0 for segment in segments)

    if segments in SIGNS_LOOKUP:
        return SIGNS_LOOKUP[segments]
    else:
        return None
