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

# videosti = "C:/Users/elleh/Downloads/IMG_0412.mp4"
videosti = '/Users/MI/Documents/SwimData/SwimCodes/temp/A.mp4'

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

# #Define the classifier
# classifier = models.vgg19(pretrained=False,progress=False)
# classifier.classifier[6] = nn.Linear(in_features=4096,out_features=len(classNames),bias=True)
# classifier.load_state_dict(torch.load("../../../SwimData/SwimCodes/classification3/models/5_0.9612403100775194.pth",
#                                       map_location=device))
# classifier = classifier.to(device)

# classtrans = transforms.Compose([ transforms.Resize((256,256)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

objectDetectorTrans = transforms.Compose([transforms.ToTensor()])


cap = cv2.VideoCapture(videosti)
t0 = time.time()
with torch.no_grad():
    master_index = 0
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
            for i in range(len(detections['labels'])):
                # print(i)
            # for detection in detections['boxes']:
                
                #we need the box points in the highRes picture. For that we need 
                #the basiskiftematrix
                box_points = detections['boxes'][i].cpu().detach().numpy()

                zoom = imagePIL.crop((box_points[0]-0, box_points[1]-0,
                                      box_points[2]+0, box_points[3]+0))
                try:
                    detectedImage = cv2.rectangle(np.array(imagePIL),(int(box_points[0]),int(box_points[1])),
                                                  (int(box_points[2]),int(box_points[3])),color=(255,255,0),
                                                  thickness=5)
                    plt.imshow(detectedImage)
                    plt.show()
                    
                    # for u in range(len(detections['labels'])):
                    plt.imshow(zoom)
                    plt.show()
                    print('Scores of objectDetector:' , detections['scores'][i])
                    if detections['scores'][i] >= 0.4:
                        zoom.save('/Users/MI/Documents/SwimData/SwimCodes/temp/A/'+str(master_index)+'_'+str(i)+'.jpg')
                    
                    master_index += 1
                    #zoom = Image.fromarray(zoom)
                    # zoom = classtrans(zoom)
                    # nul = torch.zeros((1,3,256,256))
                    # nul[0] = zoom
                    # nul = nul.to(device)
                    # outputs = classifier(nul)
                    # _, preds = torch.max(outputs, 1)
                    # print("I predict: ",classNames[preds],'with a confidence of:',detections['scores'])
                    
                except ValueError:
                    print("contains edge of image, wont work")
            print("\n")
    
    
