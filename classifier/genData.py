from xml.dom import minidom
import os
import matplotlib.pyplot as plt
import matplotlib
import random
import numpy as np
#%%
basedir = "D:/swimcamD/ObjectDetection/data/validation"

framelist = os.listdir(basedir+"/images")
annotlist = os.listdir(basedir+"/annotations")

for i in range(len(annotlist)):
    print(annotlist[i])
    mydoc = minidom.parse(basedir+"/annotations/"+annotlist[i])
    try:
        xmin = mydoc.getElementsByTagName('xmin')
        xmin = xmin[0].firstChild.data
        xmin = int(int(xmin)/400*1920)
        
        ymin = mydoc.getElementsByTagName('ymin')
        ymin = ymin[0].firstChild.data
        ymin = int(int(ymin)/400*1080)
        
        xmax = mydoc.getElementsByTagName('xmax')
        xmax = xmax[0].firstChild.data
        xmax = int(int(xmax)/400*1920)
        
        ymax = mydoc.getElementsByTagName('ymax')
        ymax = ymax[0].firstChild.data
        ymax = int(int(ymax)/400*1080)
        
        klasse = annotlist[i].split("_")[1]
        billed = plt.imread("D:/swimcamD/1080/"+klasse+"/"+
                            "1080_"+klasse+"_"+annotlist[i].split("_")[2][:-3]+"jpg")
        zoom = billed[ymin:ymax,xmin:xmax,:]
        print(ymax-ymin,ymax-ymin)
        matplotlib.image.imsave("D:/swimcamD/Classifier/validation/"+
                                klasse+"/"+klasse+str(i)+".jpg",zoom)
        
    except IndexError:
        print("ingen bounding box")
    
#%%
#Få nogle billeder ned i false

n = 200
billedliste = os.listdir("D:/swimcamD/1080/false")

for i in range(n):
    bredde = int(np.random.normal(100,3))
    højde = int(np.random.normal(100,3))
    xmin = int(random.choice(np.arange(1920)))
    ymin = int(random.choice(np.arange(1080)))
    
    billed = random.choice(billedliste)
    billed = plt.imread("D:/swimcamD/1080/false/"+billed)
    zoom = billed[ymin:ymin+højde,xmin:xmin+bredde]
    matplotlib.image.imsave("D:/swimcamD/Classifier/data/train/false/"+"false"+str(i+1800)+".jpg",zoom)
    