#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from __future__ import print_function

import cv2
# import numpy as np
import os
# import matplotlib.pyplot as plt

# homemade packages
# from swimcam import load_video #,scan_aruco



# Change folder
os.chdir("../../../SwimData/Swimcodes/taggedVideos")

files=[]
for file in os.listdir():
    if  '.mp4' in str(file)[-4:] or \
        '.MP4' in str(file)[-4:]:
        files.append(str(file))


#%%
for video in files:
# Opening of video file
# video='A_1.mp4'
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      # try:
        success,image = vidcap.read()
        if success == False:
            pass
        else:
            cv2.imwrite('../taggedPics/'+video[:3]+"_%d.jpg" % count, image)     # save frame as JPEG file
        
        # elif cv2.waitKey(10) == 27:                     # exit if Escape is hit
        #     break
        count += 1
      # except error:
      #     print('frame',count,'skipped')