# -*- coding: utf-8 -*-
"""
https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0 - распознавание полос движения. Только прямо

D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\solidWhiteRight.mp4
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\solidYellowLeft.mp4
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\project_video.mp4
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\challenge_video.mp4
D:\\ComputerVision\\_robocar\\laneDetection\\_verification\\vid\\challenge.mp4

"""

import cv2
import numpy as np
import sys

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#used below
def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

#thick red lines 
def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    global first_frame
    global cache
    
    y_global_min = img.shape[0] #min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    l_slope, r_slope = [],[]
    l_lane,r_lane = [],[]
    det_slope = 0.4
    α =0.2 

    for line in lines:
        #1
        for x1,y1,x2,y2 in line:
            slope = get_slope(x1,y1,x2,y2)
            if slope > det_slope:
                r_slope.append(slope)
                r_lane.append(line)
            elif slope < -det_slope:
                l_slope.append(slope)
                l_lane.append(line)
        #2
        y_global_min = min(y1,y2,y_global_min)
    
    # to prevent errors in challenge video from dividing by zero
    if((len(l_lane) == 0) or (len(r_lane) == 0)):
        print ('no lane detected')
        return 1
        
    #3
    l_slope_mean = np.mean(l_slope,axis =0)
    r_slope_mean = np.mean(r_slope,axis =0)
    l_mean = np.mean(np.array(l_lane),axis=0)
    r_mean = np.mean(np.array(r_lane),axis=0)
    
    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1
    
    #4, y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])
    
    #5, using y-extrema (#2), b intercept (#4), and slope (#3) solve for x using y=mx+b
    # x = (y-b)/m
    # these 4 points are our two lines that we will pass to the draw function
    l_x1 = int((y_global_min - l_b)/l_slope_mean) 
    l_x2 = int((y_max - l_b)/l_slope_mean)   
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)
    
    #6
    if l_x1 > r_x1:
        l_x1 = int((l_x1+r_x1)/2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1 ) + l_intercept)
        r_y1 = int((r_slope_mean * r_x1 ) + r_intercept)
        l_y2 = int((l_slope_mean * l_x2 ) + l_intercept)
        r_y2 = int((r_slope_mean * r_x2 ) + r_intercept)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max
      
    current_frame = np.array([l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2],dtype ="float32")
    
    if first_frame == 1:
        next_frame = current_frame        
        first_frame = 0
    else :
        prev_frame = cache
        a = (1-α)*prev_frame
        b = α*current_frame
        try:          
            next_frame = a+b
        except Exception:
            next_frame = current_frame
             
    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]),int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]),int(next_frame[7])), color, thickness)
    
    cache = next_frame
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    try:
        draw_lines(line_img,lines)
    except Exception:
        print("some linedraw error")
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):
    
    global first_frame
    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    kernel_size = 5
    gauss_gray = gaussian_blur(mask_yw_image,kernel_size)

    #same as quiz values
    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(gauss_gray,low_threshold,high_threshold)

    imshape = image.shape
    lower_left = [imshape[1]/9,imshape[0]]
    lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
    top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
    top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    roi_image = region_of_interest(canny_edges, vertices)

    #rho and theta are the distance and angular resolution of the grid in Hough space
    #same values as quiz
    rho = 4
    theta = np.pi/180
    #threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 30
    min_line_len = 100
    max_line_gap = 180
    #my hough values started closer to the values in the quiz, but got bumped up considerably for the challenge video

    line_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    return result

filepath = sys.argv[1]
 
cap = cv2.VideoCapture(filepath)

first_frame = 1

pauseMode = False

while cap.isOpened():
    if (not pauseMode):
        ret, frame = cap.read()
        if (not ret):
            break
        
        prcframe = process_image(frame)
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