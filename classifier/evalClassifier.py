import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.1,hue=0.1),
        transforms.RandomRotation(180),
        transforms.RandomPerspective(p=0.1),
        transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = "../../../SwimData/SwimCodes/classification"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=False, num_workers=0)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

#%%
def visualize_eval(model, num_images=6):
    was_training = model.training
    model.eval()



    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            fig = plt.figure()
            images_so_far = 0

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]])+
                             "\ntrue class: {}".format(class_names[labels[j]]))
                imshow(inputs.cpu().data[j])
            plt.show()

                
        model.train(mode=was_training)
        
def imshow_norm(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 

#model = torch.load("../../classifier/Cmodels/3.pt")
model_conv = torchvision.models.vgg19(pretrained=False,progress=False)
model_conv.classifier[6] = nn.Linear(in_features=4096,out_features=len(class_names),bias=True)
model_conv.load_state_dict(torch.load("../../../SwimData/SwimCodes/classification_genData/models/0batch.pth"))
model_conv = model_conv.to(device)

visualize_eval(model_conv,num_images=4)

#test inference time

#starttime = time.time()
#with torch.no_grad():
#    for inputs, labels in dataloaders["val"]:
#        inputs = inputs.to(device)
#        labels = labels.to(device)
#        outputs = model(inputs)
#        _, preds = torch.max(outputs, 1)
#time_spent = time.time()-starttime
#print(dataset_sizes["val"]/time_spent," fps")

        