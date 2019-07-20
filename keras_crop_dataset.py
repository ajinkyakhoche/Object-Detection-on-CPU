'''
@author: khoche@kth.se
This script tries to crop images from Alvin's dataset such that only objects of interest are visible. 
Since as input to keras we use 32x32x3 (similar to cifar-10), reducing our images to small size
means the features are too small to be detected.
'''

from __future__ import print_function
import cv2 as cv
import argparse

#import cv
import numpy as np 
import os
import glob

PATH_DATA = "../image_dataset-master/"

PATH_WRITE = "../image_dataset-master-cropped/"


max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

## [low]
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
## [low]

## [high]
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
## [high]

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera devide number.', default=0, type=int)
args = parser.parse_args()

## [cap]
cap = cv.VideoCapture(args.camera)
## [cap]

## [window]
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
## [window]

## [trackbar]
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
## [trackbar]


def morphOpen(image):
    # define structuring element
    # take 5% of least dimension of image as kernel size
    kernel_size = min(5, int(min(image.shape[0],image.shape[1])*0.05))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel_size,kernel_size))
    #kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return opening


for dirname in os.listdir(PATH_DATA):
    #if dirname == 'Blue Cube':  
    count = 0
    hsv_data = []
    image_name_data =[]

    if dirname == 'Blue Cube':
        continue
    if dirname == 'Red Cylinder':
        continue
    if dirname == 'Yellow Ball':
        continue
    if dirname == 'Yellow Cube':
        continue
    if dirname == 'Purple Cross':
        continue
    if dirname == 'Blue Triangle':
        continue

    for file in glob.glob(PATH_DATA + dirname + "/*.jpg"):
        if count <=100:
            count = count +1
        else:
            f1 = open(PATH_WRITE + dirname+'/hsv_data_1.txt', 'w+')
            f1.write('%s'%hsv_data)
            f1.close()

            print('wrote color data for '+dirname)

            f2 = open(PATH_WRITE + dirname+'/image_name_data_1.txt', 'w+')
            f2.write('%s'%image_name_data)
            f2.close()

            break

        image = cv.imread(file)

        # cv.imshow('Image', image)
        # if cv.waitKey(0) & 0xFF == ord('y'):
        #     print('CROP IT')
        # elif cv.waitKey(0) & 0xFF == ord('n'):
        #     print('DONT TOUCH!')


        #frame = cv.imread('../ras_objects.jpg')
        while True:
            ## [while]
            #ret, frame = cap.read()

            if image is None:
                break
            else:
                frame = np.copy(image)

            frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
            ## [while]
            frame_morph = morphOpen(frame_threshold)

            # find contours
            _, contours, _ = cv.findContours(frame_morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
            if contours.__len__() == 0:
                count = count -1
                print('NO CONTOURS FOUND, GETTING NEXT IMAGE!')
                break
            else:    
                #find the biggest area
                c = max(contours, key = cv.contourArea)

                #bounding rect
                xx,yy,w,h = cv.boundingRect(c)

                #expand Bounding box
                increment_param = 0.1

                tl_x = max(0, int(xx - increment_param*w))
                tl_y = max(0, int(yy - increment_param*h))
                br_x = min(image.shape[1], int(xx + w + increment_param*w))
                br_y = min(image.shape[0], int(yy + h + increment_param*h))

                # draw the book contour (in green)
                cv.rectangle(frame,(tl_x,tl_y),(br_x,br_y),(0,255,0),2)
                
                # Extract ROI with expanded Bounding box            
                bBox_img = image[tl_y:tl_y+(br_y-tl_y), tl_x:tl_x+(br_x-tl_x)]
                
                ## [show]
                cv.imshow(window_capture_name, cv.resize(frame, (640,480)))
                cv.imshow(window_detection_name, cv.resize(frame_threshold, (640,480)))
                ## [show]

                
                key = cv.waitKey(30)
                if key == ord('y') or key == 27:
                    cv.imwrite(PATH_WRITE + dirname + '/'+file.split('/')[-1], bBox_img)
                    print(str(count) + ':   CROPPED AND SAVED!')

                    # Extract hsv and image name data (for future use)
                    hsv_data.append([low_H,low_S,low_V,high_H,high_S,high_V])
                    image_name_data.append(file.split('/')[-1])
                    
                    break
                elif key == ord('n'):
                    count = count -1
                    print('DONt TOUCH!!!!!!!')
                    
                    # Extract hsv and image name data (for future use)
                    hsv_data.append([low_H,low_S,low_V,high_H,high_S,high_V])
                    image_name_data.append(file.split('/')[-1])
                    
                    break


