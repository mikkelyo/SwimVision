#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from __future__ import print_function

import cv2
import numpy as np
# import skvideo.io
from cv2 import aruco
import os
import matplotlib.pyplot as plt

# homemade packages
# from swimcam import load_video #,scan_aruco



# Change folder
# os.chdir('/Users/MI/Downloads/swimcam')
os.chdir('/Users/MI/Downloads/swimcam/bench')
# os.chdir('/Users/MI/Downloads/swimcam/okt8/5x5')
# os.chdir('/Users/MI/Downloads/swimcam/vids_from_oct2/aruco57')

files=[]
for file in os.listdir():
    if  '.mp4' in str(file)[-4:] or \
        '.MP4' in str(file)[-4:]:
        files.append(str(file))


#%%
# =============================================================================
# Scan single video file
# =============================================================================
#Video to be scanned:
# video = files[0]   
# video = 'aruco57_1.mp4'
# video = 'trim_oct7.mp4' 
video = '2okt_short.mp4'
# video = '4_5x5.mp4'
# video = 'oppe_5_5x5.mp4'
# video = '2_6x6.mp4'
# video = 'aruco57_1.mp4'
# Array for succesful scans
scans = []
    

# ArUco scanner parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
parameters =  aruco.DetectorParameters_create() 
# parameters.minMarkerPerimeterRate = 0.02   #These two seem to make matters worse                     
# parameters.maxMarkerPerimeterRate = 4.0  #These two seem to make matters worse                                                 
parameters.polygonalApproxAccuracyRate = 0.15 #0.16 er sweetspot for close_up_swimmer.mp4
parameters.adaptiveThreshWinSizeMax = 11       #30 good
parameters.adaptiveThreshWinSizeMin = 3         # 3 good
parameters.adaptiveThreshWinSizeStep = 1        #1 good
# parameters.perspectiveRemoveIgnoredMarginPerCell = 0.4 #0.13 default

# Background subtraction
backSub = cv2.createBackgroundSubtractorKNN() #10000 works 
# backSub.setHistory(1) #sets how many frames the BGS remembers
# backSub.setDist2Threshold(10000) #10000 was good earlier
backSub.setkNNSamples(100) #how many frames it requires to make mask

# Background subtraction
# backSub = cv2.createBackgroundSubtractorMOG2() #10000 works # 


# Opening of video file
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(video))
if not capture.isOpened:
    print('Unable to open')
    exit(0)
    
while True:
    ret, frame = capture.read()
    if frame is None:
        break
   
    # Configure mask/background sub
    mask = backSub.apply(frame)
    bwmask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    res = cv2.bitwise_and(frame,bwmask)
        
    # Frame counter top left - works questionably
    # cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    # cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    

    # Video plotting
    frame_in_question = res

    cv2.imshow('Frame', frame_in_question)
    cv2.imshow('FG Mask', frame_in_question)
    keyboard = cv2.waitKey(30)   #30 default
    if keyboard == 'q' or keyboard == 27:
        break
    
    # Detection
    scanned_frame = frame_in_question
    corners , ids, _ = aruco.detectMarkers(scanned_frame, aruco_dict, parameters=parameters)
    try:
        if ids is not None and ids != [[668]]:    
            print('\n ========================== frame: ',str(capture.get(cv2.CAP_PROP_POS_FRAMES)),'============================== \n')
            print('id: ' , ids[0][0], 'corners: ' ,corners)
            plt.figure("cornerpoints")
            plt.title(ids)
            plt.imshow(scanned_frame)
            plt.scatter(corners[0][:,:,0],corners[0][:,:,1],color="r",s=4)
            plt.show()
            scans.append(ids)
    except ValueError:
            print('========================== frame: ',str(capture.get(cv2.CAP_PROP_POS_FRAMES)),'============================== \n')
            print('2 ids detected: ' , ids, 'corners: ' ,corners)
            plt.figure("cornerpoints")
            plt.title(ids)
            plt.imshow(scanned_frame)
            plt.scatter(corners[0][:,:,0],corners[0][:,:,1],color="r",s=4)
            plt.show()
            scans.append(ids)

from scipy.stats import mode
try:
    print('Most common id found was:', mode(scans),'in',video)
    print('the ids found put in an array are:',scans)
except ValueError:
    print("\n I'm stupid and can't calculate mode(scans) when one entry has a length of 2")
except IndexError:
    print('No codes found')
#%%

# Temporary solution to save id array here when running a whole folder, so it doesnt overwrite for every iteration
scans = []

#%%
# =============================================================================
# Scan whole folder
# =============================================================================
#Video to be scanned:

for file in files:
    video = file

#put code

     