import os
from matplotlib import image as Image
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import random


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

staRCNN = torch.load("D:/swimcamD/ObjectDetection/models/11_12.311.pt",
                   map_location=device)
staRCNN.eval()

WorldClassifier = torch.load("C:/Users/elleh/Documents/swimcam2/WorldClassifier.pt",
                             map_location=device)

WorldClassifier.eval()

os.chdir("D:/swimcamD/ObjectDetection/data/validation/images")
filelist = os.listdir()
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
t0 = time.time()
for i in range(13,len(filelist)):
    print(filelist[i])
    opdelinger = filelist[i].split("_")
    imagePIL = Image.open(filelist[i])
    image = staRtrans(imagePIL)
    image_z = torch.zeros(1,3,400,400)
    image_z[0] = image
    image_z.to(device)
    detections = staRCNN(image_z)[0]["boxes"]
    print(detections)
    t1 = time.time()
    print(t1-t0)
    t0=t1
    for detection in detections:
        box_points = detection.detach().numpy()
        print(box_points)
        box_points = basisskifte @ np.array(box_points)
        high_res = plt.imread("D:/swimcamD/1080/"+opdelinger[1] + "/1080_" + opdelinger[1]+"_"+opdelinger[2])
        zoom = high_res[int(box_points[1]):int(box_points[3]),int(box_points[0]):int(box_points[2])]
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
            print(preds)
            
        except ValueError:
            print("tomt zoom")
    print("\n")