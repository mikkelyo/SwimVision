import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import datasets, models, transforms
import os
from PIL import Image
from xml.dom import minidom
import matplotlib.pyplot as plt
import numpy as np
import time 
from trainObjectDetection import SwimSet
from trainObjectDetection import mincollate
from trainObjectDetection import imshow


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 2 
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("../../../SwimData/GeoCodes/objectDetection/models/64_2.642.pth",map_location=device))
model.eval()
model.to(device)

tran = transforms.Compose([transforms.ToTensor()])

dataset_validation = SwimSet("../../../SwimData/GeoCodes/objectDetection/val",tran)
dataloader_validation = torch.utils.data.DataLoader(
 dataset_validation, batch_size=2, shuffle=True, num_workers=0,
 collate_fn=mincollate)

start = time.time()
batch_count = 1
for images,targets in dataloader_validation:
    print(targets)
    with torch.no_grad():
        if images != None:
            output = model(images)
            for i in range(len(images)):
                imshow(images[i])
                try:
                    punkt = output[i]["boxes"][0]
                    plt.scatter([punkt[0].item(),punkt[2].item()],
                            [punkt[1].item(),punkt[3].item()],color="r",s=4)
                    plt.show()
                except IndexError:
                    plt.show()
                print(output[i],"\n")
                
                #loop all detections
                for j in range(len(output[i]["boxes"])):
                    bbpred = output[i]["boxes"][j]
                    bbgt = targets[i]["boxes"][j]
                    xA = max(bbpred[0].item(), bbgt[0].item())
                    yA = max(bbpred[1].item(), bbgt[1].item())
                    xB = min(bbpred[2].item(), bbgt[2].item())
                    yB = min(bbpred[3].item(), bbgt[3].item())
                    
                    interArea = (xB - xA) * (yB - yA)
                    
                    bbpredArea = (bbpred[2].item() - bbpred[0].item()) * (bbpred[3].item() - bbpred[1].item())
                    bbgtArea = (bbgt[2].item() - bbgt[0].item()) * (bbgt[3].item() - bbgt[1].item())
                    
                    iou = interArea / float(bbpredArea + bbgtArea - interArea)
                    
                    
                    
                    
                    
            
            """The underneath can be commented in if one wishes to test inference speed """
            # nu = time.time()
            # print((2*batch_count)/(nu-start))
            # batch_count += 1
            
