# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 21:38:58 2017

@author: Илья
"""

import pickle
import cv2
import glob
import numpy as np
import matplotlib.image as mpimg

images = glob.glob('camera_cal/calibration*.jpg')

# store chessboard coordinates
chess_points = []
# store points from transformed img
image_points = []

# board is 6 rows by 9 columns. each item is one (xyz) point 
# remember, only care about inside points. that is why board is 9x6, not 10x7
chess_point = np.zeros((9*6, 3), np.float32)
# z stays zero. set xy to grid values
chess_point[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

for image in images:
    
    img = mpimg.imread(image)
    
    img = cv2.resize(img, (640, 480)) 
    
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    
    # returns boolean and coordinates
    success, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    if success:
        image_points.append(corners)
        #these will all be the same since it's the same board
        chess_points.append(chess_point)
    else:
        print('corners not found {}'.format(image))
        
img_size = (img.shape[1], img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(chess_points, image_points, img_size, None, None)

camera = {}
camera["mtx"] = mtx
camera["dist"] = dist
camera["imagesize"] = img_size
pickle.dump(camera, open("camera_matrix_640x480.pkl", "wb"))



