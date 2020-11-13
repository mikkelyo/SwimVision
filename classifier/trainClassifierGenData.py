from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
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
from genSwimCodes import GauBlur, BackGround, convert_to_rgb
#%%
# Data augmentation and normalization for training
# Just normalization for validation

# With this format, we train on generated data, and validate on real data.
data_transforms = {
    'train': transforms.Compose([transforms.Resize((256,256)),
                             # GauBlur(1),
                             # transforms.RandomRotation(180),
                             # transforms.RandomGrayscale(),
                             # transforms.RandomHorizontalFlip(),
                             # transforms.RandomPerspective(p=0.8),
                             # BackGround(1,"../../../SwimData/SwimCodes/classification/train/False"),
                             # GauBlur(0.5),
                             # convert_to_rgb(),
                             transforms.ToTensor()
                             ]),
    
    'val': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = "../../../SwimData/SwimCodes/classification_genData"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        batch_count = 1
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('Current phase:',phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
        
                with torch.set_grad_enabled(phase == 'train'):
                    # outputs from model
                    outputs = model(inputs)

                    # output with highest probability picked out
                    _, preds = torch.max(outputs, 1)
                    print('Predictions for batch of 32 inputs:',preds)
                    
                    
                    # loss is calculated by comparing with labels
                    loss = criterion(outputs, labels)
                    print('Loss from',batch_count,'batch:',loss.item())
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # Saves per batch if you want immediate results
                # print('Saving model...')
                # torch.save(model.state_dict(),"../../../SwimData/SwimCodes/classification_genData/models/"+str(epoch)+'batch'+".pth")
                if batch_count%50 ==0:
                    print('Batch',batch_count,'completed succesfully')
                batch_count += 1
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Saving model...')
                torch.save(model.state_dict(),"../../../SwimData/SwimCodes/classification_genData/models/"+str(epoch)+'_'+str(epoch_acc.item())+".pth")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
    
    
#%%
model_conv = torchvision.models.vgg19(pretrained=True,progress=False)

print(model_conv)

for param in model_conv.parameters():
    param.requires_grad = False


# Parameters of newly constructed modules have requires_grad=True by default

#num_ftrs = model_conv.classifier.in_features
#model_conv.fc = nn.Linear(num_ftrs, 3)
model_conv.classifier[6] = nn.Linear(in_features=4096,out_features=len(class_names),bias=True)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.classifier[6].parameters(), lr=0.001, momentum=0.9)
#optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.0005)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

if __name__ == "__main__":
    print(model_conv)
    
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=1500)




