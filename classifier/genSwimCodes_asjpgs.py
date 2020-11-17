import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import PIL
import cv2
import random
from PIL import Image



class GauBlur(object):
    def __init__(self,p):
        self.kernelsize = int(np.random.normal(27*5,5*5))
        self.sd = abs(np.random.normal(7*1.5,2*1.5))
        self.p = p
        
    
    def __call__(self,img):
        if random.random() < self.p:
            gauker = cv2.getGaussianKernel(self.kernelsize,self.sd)
            slør = cv2.filter2D(np.array(img), -1, gauker)
            slør = Image.fromarray(slør)
            return slør
        else:
            return img
        
class convert_to_rgb(object):
    def __init__(self):
        pass
        # self.object.convert('RGB')
    def __call__(self,img):
        rgb_img = img.convert('RGB')
        return rgb_img
        


# class HoriBlur:
#     def __init__(self,p):
#         self.kernelsize = int(abs(np.random.normal(19,3)))
#         self.p = p
#     def __call__(self,img):
#         if random.random() < self.p:
#             kernel = np.zeros((self.kernelsize,self.kernelsize))
#             #kernel[int((self.kernelsize-1)/2),:] = np.random.normal(1/5,2,self.kernelsize)
#             kernel[int((self.kernelsize-1)/2),:] = 1/20
#             print(kernel.shape)
#             print(kernel)
#             slør = cv2.filter2D(np.array(img), -1, kernel)
#             slør = Image.fromarray(slør)
#             return slør
#         else:
#             return img
        
# class VertiBlur:
#     def __init__(self,p):
#         self.kernelsize = int(abs(np.random.normal(19,7)))
#         self.p = p
#     def __call__(self,img):
#         if random.random() < self.p:
#             kernel = np.zeros((self.kernelsize,self.kernelsize))
#             kernel[:,int((self.kernelsize-1)/2)] = 1/5
#             slør = cv2.filter2D(np.array(img), -1, kernel)
#             slør = Image.fromarray(slør)
#             return slør
#         else:
#             return img


        
# class Blur:
#     def __init__(self,p,img):
#         self.p = p
#         self.img = img
    
#     def GauBlur(self):
#         kernelsize = int(np.random.normal(27,5))
#         sd = abs(np.random.normal(7,2))
#         gauker = cv2.getGaussianKernel(kernelsize,sd)
#         slør = cv2.filter2D(np.array(self.img), -1, gauker)
#         slør = Image.fromarray(slør)
#         return slør
    
#     def HoriBlur(self):
#         kernelsize = int(abs(np.random.normal(19,7)))
#         kernel = np.zeros((kernelsize,kernelsize))
#         kernel[int((kernelsize-1)/2),:] = 1/5
#         slør = cv2.filter2D(np.array(self.img), -1, kernel)
#         slør = Image.fromarray(slør)
#         return slør
    
#     def VertiBlur(self):
#         kernelsize = int(abs(np.random.normal(19,7)))
#         kernel = np.zeros((kernelsize,kernelsize))
#         kernel[:,int((kernelsize-1)/2)] = 1/5
#         slør = cv2.filter2D(np.array(self.img), -1, kernel)
#         slør = Image.fromarray(slør)
#         return slør
    
    # def __call__(self):
    #     if random.random() < self.p:
    #         transliste = ["G","H","V"]
    #         random.shuffle(transliste)
    #         for trans in transliste:
    #             if trans == "G":
    #                 self.img = GauBlur(self)
    #             elif trans == "H":
    #                 self.img = HoriBlur(self)
    #             elif trans == "V":
    #                 self.img = VertiBlur(self)
    #         return self.img
    #     else:
    #         return self.img
        
class BackGround(object):
    def __init__(self,p,path):
        self.path = path
        self.p = p

    def __call__(self,img):
        if random.random() < self.p:
            background = random.choice(os.listdir(self.path))
            try:
                background = Image.open(self.path+"/"+background)
            except OSError:
                return img
            background = background.resize((256,256))
            background = background.convert("RGBA")
            
            img = img.resize((350,350))
            img = img.convert("RGBA")
            
            points = np.linspace(-75,0,1)
            x = int(random.choice(points))
            y = int(random.choice(points))
            background.paste(img,(x,y),img)
            return background
        else:
            return img



trans2 = transforms.Compose([transforms.Resize((256,256)),
                             GauBlur(1),
                             transforms.RandomRotation(180),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomPerspective(p=0.8),
                             transforms.ColorJitter(brightness=0.5),
                             BackGround(1,"../../../SwimData/SwimCodes/classification/train/False"),
                             GauBlur(0.5),
                             transforms.Resize((15,15)),
                             transforms.Resize((256,256)),
                             ])

#BackGround(0.9,"../../background")

# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize((256,256)),
#         transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.5,hue=0.3),
#         transforms.RandomRotation(180),
#         transforms.RandomPerspective(p=0.1),
#         transforms.RandomGrayscale(),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize((256,256)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }



image = PIL.Image.open("../../../SwimData/SwimCodes/SwimCodes_pngs/A/SwimCode1_transparent.png")

for i in range(500):
    gen_image = trans2(image)
    gen_image = gen_image.convert('RGB')
    # plt.imshow(gen_image)
    # plt.show()
    if i%50==0:
        print(i,'images generated')
    gen_image.save('/Users/MI/Documents/GitHub/SwimVision/classifier/generated_swimcodes/D/'+'D'+str(i)+'.jpg')












