import os
from matplotlib import image as Image
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn
import cv2 


#define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#define classnames
classNames = ["A","B","C","D","False"]

#Define the object detector model as objectDetector

objectDetector = models.detection.fasterrcnn_resnet50_fpn()
num_classes = 2 
in_features = objectDetector.roi_heads.box_predictor.cls_score.in_features
objectDetector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
objectDetector.load_state_dict(torch.load("../../../SwimData/SwimCodes/objectDetection/models/RCNN_13nov.pth",
                                          map_location=device))
objectDetector.eval()
objectDetector.to(device)

#Define the classifier
classifier = models.vgg19(pretrained=False,progress=False)
classifier.classifier[6] = nn.Linear(in_features=4096,out_features=len(classNames),bias=True)
classifier.load_state_dict(torch.load("../../../SwimData/SwimCodes/classification/models/6_0.8320413436692506.pth",
                                      map_location=device))
classifier = classifier.to(device)


valPath = "../../../SwimData/SwimCodes/objectDetection/val/images"

filelist = os.listdir(valPath)

random.shuffle(filelist)

basisskifte = np.array([[1920/400,0,0,0],
                       [0,1080/400,0,0],
                       [0,0,1920/400,0],
                       [0,0,0,1080/400]])

classtrans = transforms.Compose([ transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

objectDetectorTrans = transforms.Compose([transforms.ToTensor()])


t0 = time.time()
with torch.no_grad():
    for i in range(len(filelist)):
        print("current file: ",filelist[i])
        #Alle filer follow the pattern resolution_className_index. We extract these using split
        splits = filelist[i].split("_")
        
        #the object detector needs the entire lowRes picture. We load that in as tensor
        #of shape (1,3,W,H)
        imagePIL = Image.open(os.path.join(valPath,filelist[i]))
        image = objectDetectorTrans(imagePIL)
        image_z = torch.zeros(1,3,image.shape[1],image.shape[2])
        image_z[0] = image
        image_z.to(device)
        
        #We make inference. Ignore everything other than "boxes"
        detections = objectDetector(image_z)[0]["boxes"]
        print("Box points: ",detections)
        t1 = time.time()
        print(t1-t0," sec")
        t0=t1
        
        #looping through all suggested boxes
        for detection in detections:
            #we need the box points in the highRes picture. For that we need 
            #the basiskiftematrix
            box_points = detection.detach().numpy()
            zoom = imagePIL.crop((box_points[0],box_points[1],box_points[2],box_points[3]))
            try:
                detectedImage = cv2.rectangle(np.array(imagePIL),(int(box_points[0]),int(box_points[1])),
                                              (int(box_points[2]),int(box_points[3])),color=(255,255,0),
                                              thickness=5)
                plt.imshow(detectedImage)
                plt.show()
                
                plt.imshow(zoom)
                plt.show()
                
                #zoom = Image.fromarray(zoom)
                zoom = classtrans(zoom)
                nul = torch.zeros((1,3,256,256))
                nul[0] = zoom
                nul = nul.to(device)
                outputs = classifier(nul)
                _, preds = torch.max(outputs, 1)
                print("I predict: ",classNames[preds])
                
            except ValueError:
                print("tomt zoom")
        print("\n")