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
os.chdir("../classifier")
from trainClassifierGenAndReal import class_names
os.chdir("../main") 

# videosti = "C:/Users/elleh/Downloads/IMG_0412.mp4"
videosti = '../../../SwimData/GeoCodes/temp/IMG_0063.MOV'
#videosti = '/Users/MI/Downloads/IMG_0067.MOV'

#define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Object Detector
#Define the object detector model as objectDetector

objectDetector = models.detection.fasterrcnn_resnet50_fpn()
num_classes = 2 
in_features = objectDetector.roi_heads.box_predictor.cls_score.in_features
objectDetector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
objectDetector.load_state_dict(torch.load("../../../SwimData/GeoCodes/objectDetection/models/64_2.642.pth",
                                          map_location=device))
objectDetector.eval()
objectDetector.to(device)


# Define the classifier

#define class_names
#class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "False"]


classifier = models.vgg19(pretrained=False,progress=False)
classifier.classifier[6] = nn.Linear(in_features=4096,out_features=len(class_names),bias=True)
classifier.load_state_dict(torch.load("../../../SwimData/GeoCodes/classifier4/models/1_1.0.pth",
                                      map_location=device))
classifier = classifier.to(device)

classtrans = transforms.Compose([ transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

objectDetectorTrans = transforms.Compose([transforms.ToTensor()])

# Defining softmax layer to get a confidence value for the classification
softmaxlayer = torch.nn.Softmax(dim=1)

cap = cv2.VideoCapture(videosti)
t0 = time.time()

# torch.no_grad() is included to save memory on the computer - prevents torch from constantly monitoring the gradient during backpropagation
with torch.no_grad():
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # plt.imshow(frame)
            # plt.show()
            
            imagePIL = Image.fromarray(frame)
            image = objectDetectorTrans(imagePIL)
            image_z = torch.zeros(1,3,image.shape[1],image.shape[2])
            image_z[0] = image
            image_z = image_z.to(device)
            
            #We make inference. Ignore everything other than "boxes"
            detections = objectDetector(image_z)[0] #["boxes"]
            print("Box points: ",detections['boxes'])
            t1 = time.time()
            print(t1-t0," sec")
            t0=t1
            for i in range(len(detections['boxes'])):
            # for detection in detections['boxes']:
                box_points = detections['boxes'][i].cpu().detach().numpy()
                zoom = imagePIL.crop((box_points[0]-20, box_points[1]-20,
                                      box_points[2]+20, box_points[3]+20))
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
                    print('OBJECT DETECTOR: Object detected with a confidence of',detections['scores'][i])
                    print("CLASSIFIER: I predict: ---",class_names[preds],'--- with a confidence of:',softmaxlayer(outputs)[0][preds])
                    
                except ValueError:
                    print("zoom is including areas outside the image, this error hasnt been fixed yet")
            print("\n")
        else:
            break
    
    
