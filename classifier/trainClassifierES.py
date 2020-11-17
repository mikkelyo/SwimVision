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
from sklearn.metrics import confusion_matrix
from genSwimCodes import GauBlur, BackGround, convert_to_rgb
#%%
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'realTrain': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'realVal': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'artTrain': transforms.Compose([transforms.Resize((256,256)),
        GauBlur(1),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(p=0.8),
        transforms.ColorJitter(brightness=0.5),
        BackGround(1,"../../../SwimData/SwimCodes/classification/train/False"),
        GauBlur(0.5),
        transforms.Resize((15,15)),
        transforms.Resize((256,256)),
        convert_to_rgb(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'artVal' : transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}
batch_size = 4
artLen = 1000

data_dir = "../../../SwimData/SwimCodes/classification3"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['realTrain', 'realVal', "artTrain", "artVal"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ['realTrain', 'realVal',"artTrain","artVal"]}

classCounts = {x:[image_datasets[x].targets.count(Class) for
              Class in np.unique(image_datasets[x].targets)]
               for x in ['realTrain', 'realVal', "artTrain", "artVal"]}

sampleWeights = {x: np.array([1.0/np.array(classCounts[x])[t] for t in image_datasets[x].targets])
                 for x in ['realTrain', 'realVal', "artTrain", "artVal"]}

Ns = {"realTrain": dataset_sizes["realTrain"],
      "realVal": dataset_sizes["realVal"],
      "artTrain": artLen,
      "artVal": dataset_sizes["artVal"]}

samplers = {x:torch.utils.data.WeightedRandomSampler(sampleWeights[x],
                                                                 Ns[x])
            for x in ["realTrain", "realVal", "artTrain", "artVal"]}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              num_workers=0, sampler=samplers[x])                                             
               for x in ["realTrain", "realVal", "artTrain", "artVal"]}
                

class_names = image_datasets['realTrain'].classes
print('Types of classes:',class_names)

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
    plt.pause(0.001) # pause a bit so that plots are updated



def confusionMatrix(dataloader):
    all_preds = []
    all_labels = []
    j = 1
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = classifier(inputs)
            _, preds = torch.max(outputs, 1)
            for i in range(len(preds)):
                all_preds.append(preds[i].item())
                all_labels.append(labels[i].item())
            print(j)
            conf = confusion_matrix(all_labels,all_preds)
            plt.imshow(conf, interpolation="nearest", cmap=plt.cm.Blues)
            plt.colorbar()
            plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
            plt.yticks(np.arange(len(class_names)), class_names)
            plt.ylabel("Actual code")
            plt.show()
            j += 1
            

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ["realTrain", "realVal", "artTrain", "artVal"]:
            batch_count = 1
            if (phase == 'realTrain' or phase == "artTrain"):
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                print(labels)
                imshow(inputs[0])
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'realTrain' or phase == "artTrain"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if (phase == 'realTrain' or phase == "artTrain"):
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if batch_count%50 ==0:
                    print('Batch',batch_count,'completed succesfully')
                batch_count += 1
            if (phase == 'realTrain' or phase == "artTrain"):
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'realVal' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                print('Saving model...')
                torch.save(model.state_dict(),os.path.join(data_dir,"models",
                           str(epoch)+"_"+str(epoch_acc.item())+".pth"))

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
classifier = torchvision.models.vgg19(pretrained=True,progress=False)

# for param in classifier.parameters():
#     param.requires_grad = False


# Parameters of newly constructed modules have requires_grad=True by default

#num_ftrs = classifier.classifier.in_features
#classifier.fc = nn.Linear(num_ftrs, 3)
classifier.classifier[6] = nn.Linear(in_features=4096,out_features=len(class_names),bias=True)



classifier = classifier.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
#optimizer_conv = optim.SGD(classifier.classifier[6].parameters(), lr=0.001, momentum=0.9)
optimizer_conv = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

if __name__ == "__main__":
    print(classifier)
    classifier = train_model(classifier, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=1500)






                   




