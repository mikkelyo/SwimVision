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
for letter in ['rc0','bc0','gc0','rc1','bc1','gc1','rf0','gf0','bf0','rf1','gf1','bf1','rt0','bt0','gt0','rt1','bt1','gt1','rl0','bl0','gl0','rl1','bl1','gl1','rs0','bs0','gs0','rs1','bs1','gs1','false']:
    
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
    
    
    
    
    
    