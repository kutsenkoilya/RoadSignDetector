# -*- coding: utf-8 -*-
"""
Распознаввтель знаков по картинке

D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\pic\\roadswsigns\\01_test.jpg
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\pic\\roadswsigns\\02_test.jpg
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\pic\\roadswsigns\\03_test.jpg
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\pic\\roadswsigns\\04_test.jpg
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\pic\\roadswsigns\\05_test.jpg
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\pic\\roadswsigns\\06_test.jpg
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\pic\\roadswsigns\\07_test.jpg
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\pic\\roadswsigns\\08_test_r.jpg
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\pic\\roadswsigns\\09_test_r.jpg
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\pic\\roadswsigns\\10_test_r.jpg
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\pic\\roadswsigns\\11_test_r.jpg
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from lib_roundSignDetector import findRoundTrafficSign
from lib_squareSignDetector import findSquareTrafficSign
import cv2
import numpy as np
import sys
import os
import imutils

filepath = 'D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\pic\\roadswsigns\\'
datasetLen = len(os.listdir(filepath))


fig = plt.figure(1, (25., 25.))
grid = ImageGrid(fig, 111, 
                 nrows_ncols=(datasetLen, 2), 
                 axes_pad=0.1,
                 )

index = 0

for source_img in os.listdir(filepath):
    print(source_img)
    img = cv2.imread(filepath+source_img, cv2.IMREAD_COLOR)
    
    #mod = findSquareTrafficSign(img,True)
    mod = findRoundTrafficSign(img,False)
    
    #cv2.imshow('modified', mod)
    #cv2.imshow('original', img)
    
    grid[index].imshow(imutils.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=500)) 
    index = index +1
    grid[index].imshow(cv2.cvtColor(mod, cv2.COLOR_BGR2RGB)) 
    index = index +1

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    
    








