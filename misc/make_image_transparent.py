#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:11:43 2020

@author: MI
"""
from PIL import Image
import os

#%%

# set letter (has to be valid)
letter = 'B'

# set folder
os.chdir('/Users/MI/Documents/SwimData/SwimCodes/classification5/artTrain/'+letter)

img = Image.open(letter+'.png')
img = img.convert("RGBA")
datas = img.getdata()

newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)

img.putdata(newData)
img.save(letter+"_transparent.png", "PNG")
#%%
# =============================================================================
# For multiple letters
# =============================================================================

# set letter (has to be valid)
for letter in ['A','B','C','D','E','F','G','H']:
    
    # set folder
    os.chdir('/Users/MI/Documents/SwimData/SwimCodes/classification5/artTrain/'+letter)
    
    img = Image.open(letter+'.png')
    img = img.convert("RGBA")
    datas = img.getdata()
    
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    
    img.putdata(newData)
    img.save(letter+"_transparent.png", "PNG")