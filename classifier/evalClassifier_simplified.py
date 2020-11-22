import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import torch.nn as nn
from PIL import Image
from genSwimCodes import GauBlur, BackGround, convert_to_rgb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ["A","B","C","D","E","F","G","H"]
classifier = torchvision.models.vgg19(pretrained=False,progress=False)
classifier.classifier[6] = nn.Linear(in_features=4096,out_features=len(class_names),bias=True)
classifier.load_state_dict(torch.load("../../../SwimData/SwimCodes/classification5/models/8_1.0.pth",
                                      map_location=device))
classifier = classifier.to(device)


root = "../../../Swimdata/SwimCodes/temp/fotografier"
filliste = os.listdir(root)
trans = transforms.Compose([
        transforms.Resize((256, 256)),
        convert_to_rgb(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

for i in range(len(filliste)):
    try:
        print(filliste[i])
        frame = Image.open(os.path.join(root,filliste[i]))
        frame = trans(frame)
        frame_z = torch.zeros((1,3,256,256))
        frame_z[0] = frame
        frame_z = frame_z.to(device)
        outputs = classifier(frame_z)
        print(outputs)
        _, preds = torch.max(outputs, 1)
        print("I predict :",class_names[preds],"\n")
    except OSError:
        pass
    
