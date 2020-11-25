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

#vi definerer device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SwimSet(object):
    def __init__(self,root,transforms):
        self.root = root
        self.transforms = transforms
        
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))
        
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,idx):
        try:
            annotation_path = os.path.join(self.root, "annotations", self.annotations[idx])
        except IndexError:
            return None, None
        annotation = minidom.parse(annotation_path)
        fileitems = annotation.getElementsByTagName('filename')
        img_path = fileitems[0].firstChild.data
        img_path = os.path.join(self.root, "images", img_path)
        img = Image.open(img_path).convert("RGB")           
        
        img = self.transforms(img)
        
        annotation = minidom.parse(annotation_path)
        items = annotation.getElementsByTagName('object')
        if len(items) == 0:
            boxes = [[0,0,img.shape[1],img.shape[2]]]
            labels = torch.zeros((1,), dtype=torch.int64)
        else:
            xmin = annotation.getElementsByTagName('xmin')
            xmin = int(xmin[0].firstChild.data)
            
            ymin = annotation.getElementsByTagName('ymin')
            ymin = int(ymin[0].firstChild.data)
            
            xmax = annotation.getElementsByTagName('xmax')
            xmax = int(xmax[0].firstChild.data)
            
            ymax = annotation.getElementsByTagName('ymax')
            ymax = int(ymax[0].firstChild.data)
        
            boxes = [[xmin,ymin,xmax,ymax]]
            labels = torch.ones((1,), dtype=torch.int64)
        
        target = {}
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        boxes = boxes.to(device)
        labels = labels.to(device)
        target["boxes"] = boxes
        target["labels"] = labels
        
            
        return img,target
    
def mincollate(data):
    images_temp,targets_temp = zip(*data)
    batch_size = len(images_temp)
    
    if None in images_temp:
        print("batch will be ignored")
        images = None
        targets = None
        return images, targets
    
    #handle images
    #Missing annotations -> ignore the batch
    images = torch.zeros((batch_size,3,images_temp[0].shape[1],
                                    images_temp[0].shape[2]))
    for i in range(batch_size):
        images[i] = images_temp[i]
    images = images.to(device)
        
    #handle targets
    targets = [{"boxes":t["boxes"],"labels":t["labels"]} for t in targets_temp]
    return images, targets

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu()
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    
def train(model,device,root,tran,batch_size=2,
          epochs=100):
    
    #define a train and a validation data loader
    dataset_train = SwimSet(os.path.join(root,"train"),tran)
    dataloader_train = torch.utils.data.DataLoader(
         dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
         collate_fn=mincollate)
    
    dataset_validation = SwimSet(os.path.join(root,"val"),tran)
    dataloader_validation = torch.utils.data.DataLoader(
        dataset_validation, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=mincollate)
    
    # construct an optimizer and a learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                      step_size=3,
                                                      gamma=0.1)
    
    for epoch in range(epochs):
        
        #training
        totalLoss = 0.0
        i = 1
        start = time.time()
        for images, targets in dataloader_train:
            #only do something if batch is not empty
            if images != None:
                optimizer.zero_grad()
                output = model(images,targets)
                loss = output["loss_classifier"]+output["loss_box_reg"]
                loss.backward()
                optimizer.step()
                totalLoss += loss.item()
                
                 #print something every ___ batch
                if i % 20 == 0:
                    print("\nloss pr batch: ",totalLoss/(i-1))
                    nu = time.time()
                    print("sek pr. batch: ",(nu-start)/(i-1))
                i += 1
                # torch.save(model.state_dict(),os.path.join(root,"models")+"/"+str(epoch)+"_"+str(round(totalLoss,3))+".pth")
        print("træning er færdig")
        print("totalLoss for train: ",totalLoss)
        print("evaling ...\n")
        
        #evaling. We use torch.no_grad() to stop the program from crashing
        with torch.no_grad():
            totalLoss = 0.0
            for images, targets in dataloader_validation:
                #only do something if batch is not empty
                if images != None:
                    optimizer.zero_grad()
                    output = model(images,targets)
                    loss = output["loss_classifier"]+output["loss_box_reg"]
                    totalLoss += loss.item()
            print("eval færdig")
            print("totalLoss for eval: ",totalLoss)
            print("\nepoke er færdig")
        lr_scheduler.step()
        torch.save(model.state_dict(),os.path.join(root,"models")+"/"+str(epoch)+"_"+str(round(totalLoss,3))+".pth")
        
if __name__ == "__main__":
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    
    tran = transforms.Compose([transforms.ToTensor()])
    
    root = "../../../SwimData/GeoCodes/objectDetection"
    
    train(model,device,root,tran)
    

    
    
    
        

    
    
        
    

    