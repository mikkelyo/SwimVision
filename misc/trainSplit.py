import os
import shutil
import numpy as np
import random

#%%
#Lave et validation datasæt
#mappeliste = os.listdir("D:/swimcamD/ObjectDetection/train/images") 
splitRatioVal = 0.2

filliste = os.listdir("../../../SwimData/GeoCodes/objectDetection/train/images")
N = len(filliste)
n = int(splitRatioVal * N)
flytteliste = random.sample(filliste,k=n)
for fil in flytteliste:
    print(fil)
    # shutil.move("D:/swimcamD2/SwimData/SwimCodes/objectDetection/train/images/"+fil,
    #              "D:/swimcamD2/SwimData/SwimCodes/objectDetection/val/images/"+fil)
    shutil.move("../../../SwimData/GeoCodes/objectDetection/train/images/"+fil,
                  "../../../SwimData/GeoCodes/objectDetection/val/images/"+fil)
    kernenavn = fil[:-4]
    try:
        # shutil.move("D:/swimcamD2/SwimData/SwimCodes/objectDetection/train/annotations/"+kernenavn+".xml",
        #              "D:/swimcamD2/SwimData/SwimCodes/objectDetection/val/annotations/"+kernenavn+".xml")
        shutil.move("../../../SwimData/GeoCodes/objectDetection/train/annotations/"+kernenavn+".xml",
                     "../../../SwimData/SwimCodes/objectDetection/val/annotations/"+kernenavn+".xml")
    except FileNotFoundError:
        print("den annotering manglede. Jeg lader bare som ingenting. Fløjte fløjte")
         
                                  