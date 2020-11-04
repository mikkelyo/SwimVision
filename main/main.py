import os
from matplotlib import image as Image
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import random

#define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Define the object detector model as staRCNN
staRCNN = torch.load("../../../SwimData/arucoOctober/objectDetection/models/11_12.311.pt",
                   map_location=device)
staRCNN.eval()

#Define the classifier
WorldClassifier = torch.load("../../../SwimData/arucoOctober/classification/models/WorldClassifier.pt",
                             map_location=device)

WorldClassifier.eval()

lowResPath = "../../../SwimData/arucoOctober/objectDetection/val/images"
highResPath = "../../../SwimData/arucoOctober/classification/1080/"
filelist = os.listdir(lowResPath)
random.shuffle(filelist)

basisskifte = np.array([[1920/400,0,0,0],
                       [0,1080/400,0,0],
                       [0,0,1920/400,0],
                       [0,0,0,1080/400]])

classtrans = transforms.Compose([ transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

staRtrans = transforms.Compose([transforms.ToTensor()])
classNames = ["5x5","6x6","False"]

t0 = time.time()
for i in range(len(filelist)):
    print("current file: ",filelist[i])
    #Alle filer follow the pattern resolution_className_index. We extract these using split
    splits = filelist[i].split("_")
    
    #the object detector needs the entire lowRes picture. We load that in as tensor
    #of shape (1,3,W,H)
    imagePIL = Image.open(os.path.join(lowResPath,filelist[i]))
    image = staRtrans(imagePIL)
    image_z = torch.zeros(1,3,400,400)
    image_z[0] = image
    image_z.to(device)
    
    #We make inference. Ignore everything other than "boxes"
    detections = staRCNN(image_z)[0]["boxes"]
    print("Box points: ",detections)
    t1 = time.time()
    print(t1-t0," sec")
    t0=t1
    
    #looping through all suggested boxes
    for detection in detections:
        #we need the box points in the highRes picture. For that we need 
        #the basiskiftematrix
        box_points = detection.detach().numpy()
        box_points = basisskifte @ np.array(box_points)
        highRes = plt.imread(highResPath+
                              splits[1] + "/1080_" + splits[1]+"_"+splits[2])
        zoom = highRes[int(box_points[1]):int(box_points[3]),int(box_points[0]):int(box_points[2])]
        try:
            plt.imshow(np.array(imagePIL))
            plt.show()
            plt.imshow(zoom)
            plt.show()
            
            zoom = Image.fromarray(zoom)
            zoom = classtrans(zoom)
            nul = torch.zeros((1,3,256,256))
            nul[0] = zoom
            nul = nul.to(device)
            outputs = WorldClassifier(nul)
            _, preds = torch.max(outputs, 1)
            print("I predict: ",classNames[preds])
            
        except ValueError:
            print("tomt zoom")
    print("\n")