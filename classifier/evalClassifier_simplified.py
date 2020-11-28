import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import torch.nn as nn
from PIL import Image
os.chdir("../misc")
from genSwimCodes import GauBlur, BackGround, convert_to_rgb
os.chdir("../classifier")
from trainClassifierGenAndReal import class_names


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#class_names = ["A","B","C","D","E","F","G","H"]
classifier = torchvision.models.vgg19(pretrained=False,progress=False)
classifier.classifier[6] = nn.Linear(in_features=4096,out_features=len(class_names),bias=True)
classifier.load_state_dict(torch.load("../../../SwimData/GeoCodes/classifier4/models/2_1.0.pth",
                                      map_location=device))
classifier = classifier.to(device)


root = "../../../Swimdata/GeoCodes/temp/fotografier"
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
        plt.imshow(frame)
        plt.show()
        frame = trans(frame)
        frame_z = torch.zeros((1,3,256,256))
        frame_z[0] = frame
        frame_z = frame_z.to(device)
        outputs = classifier(frame_z)
        _, preds = torch.max(outputs, 1)
        _, preds2 = outputs.topk(5)
        print("I predict :",class_names[preds],"\n")
        print("I also predict: ",[class_names[preds2[0][i]] for i in range(5)])
    except OSError:
        pass
    
