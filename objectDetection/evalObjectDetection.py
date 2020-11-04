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
from trainfasterrcnn import SwimSet
from trainfasterrcnn import mincollate
from trainfasterrcnn import imshow

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torch.load("../../../SwimData/arucoOctober/objectDetection/models/11_12.311.pt",
                   map_location=device)

model.eval()

tran = transforms.Compose([transforms.ToTensor()])

dataset_validation = SwimSet("../../../SwimData/arucoOctober/objectDetection/val",tran)
dataloader_validation = torch.utils.data.DataLoader(
 dataset_validation, batch_size=2, shuffle=True, num_workers=0,
 collate_fn=mincollate)

start = time.time()
i = 1
for images,targets in dataloader_validation:
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
            nu = time.time()
            print(output[i],"\n")
            #print((2*i)/(nu-start))
            i += 1
        
