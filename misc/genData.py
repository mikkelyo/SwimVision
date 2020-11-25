from xml.dom import minidom
import os
import matplotlib.pyplot as plt
import matplotlib
import random
import numpy as np
#%%
basedir = "../../../SwimData/GeoCodes/objectDetection/train"

framelist = os.listdir(basedir+"/images")
annotlist = os.listdir(basedir+"/annotations")

for i in range(len(annotlist)):
    print(annotlist[i])
    mydoc = minidom.parse(basedir+"/annotations/"+annotlist[i])
    try:
        xmin = mydoc.getElementsByTagName('xmin')
        xmin = xmin[0].firstChild.data
        #xmin = int(int(xmin)/400*1920)
        xmin = int(xmin)
        
        ymin = mydoc.getElementsByTagName('ymin')
        ymin = ymin[0].firstChild.data
        #ymin = int(int(ymin)/400*1080)
        ymin = int(ymin)
        
        xmax = mydoc.getElementsByTagName('xmax')
        xmax = xmax[0].firstChild.data
        #xmax = int(int(xmax)/400*1920)
        xmax = int(xmax)
        
        ymax = mydoc.getElementsByTagName('ymax')
        ymax = ymax[0].firstChild.data
        #ymax = int(int(ymax)/400*1080)
        ymax = int(ymax)
        
        billedFilNavn = mydoc.getElementsByTagName('filename')
        billedFilNavn = billedFilNavn[0].firstChild.data
        print(billedFilNavn)
        
        klasse = annotlist[i].split("_")[0]
        # billed = plt.imread("D:/swimcamD/1080/"+klasse+"/"+
        #                     "1080_"+klasse+"_"+annotlist[i].split("_")[2][:-3]+"jpg")
        billed = plt.imread(basedir+"/images/"+billedFilNavn)
        zoom = billed[ymin:ymax,xmin:xmax,:]
        print(ymax-ymin,ymax-ymin)
        # matplotlib.image.imsave("../../../SwimData/SwimCodes/classification/train/"+
        #                         klasse+"/"+klasse+str(i)+".jpg",zoom)
        plt.imsave("../../../SwimData/GeoCodes/classifier/realTrain/"+
                                 klasse+"/"+klasse+str(i)+".jpg",zoom)
    except IndexError:
        print("ingen bounding box")
    
#%%
#Få nogle billeder ned i false
mappe = '/Users/MI/Documents/SwimData/GeoCodes/classifier/temp'
n = 50
billedliste = os.listdir(mappe)

for i in range(n):
    bredde = int(np.random.normal(100,3))
    højde = int(np.random.normal(100,3))
    xmin = int(random.choice(np.arange(1920)))
    ymin = int(random.choice(np.arange(1080)))
    
    billed = random.choice(billedliste)
    billed = plt.imread('/Users/MI/Documents/SwimData/GeoCodes/classifier/temp'+"/"+billed)
    zoom = billed[ymin:ymin+højde,xmin:xmin+bredde]
    matplotlib.image.imsave("/Users/MI/Documents/SwimData/GeoCodes/classifier/realTrain/false/"+"false"+str(i+3900)+".jpg",zoom)
    