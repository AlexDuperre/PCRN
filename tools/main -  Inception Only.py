import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from DrawDataset import MyDataset
from Inception4 import inceptionv4

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

###############################################################################################################################
# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# Hyper parameters
num_epochs = 20
num_classes = 55
batch_size = 256

learning_rate = 0.00001
validationRatio = 0.1

# train data
class MyTransform(object):
    def __call__(self,tensor):
        tensor = torch.abs(tensor-1)
        return tensor[0,:,:].unsqueeze(0)

transformations = transforms.Compose([torchvision.transforms.Resize((300,300)),
                                     transforms.ToTensor(),
                                      MyTransform()])


dataset = MyDataset('../../../DATA/alex/ShapeNetCoreV2 - Depth/', transform= transformations)


# sending to loader
torch.manual_seed(0)
indices = torch.randperm(len(dataset))
train_indices = indices[:len(indices) - int((validationRatio) * len(dataset))]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_indices = indices[len(indices) - int(validationRatio * len(dataset)):]
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

# Dataset
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           sampler = train_sampler,
                                           shuffle=False,
                                           num_workers=1)
valid_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           sampler = valid_sampler,
                                           shuffle=False,
                                           num_workers=1)


# for i, (images, labels) in enumerate(train_loader):
#     plt.imshow(images[1].numpy())
#     plt.show()
#     print(labels)
#     break

model = inceptionv4(num_classes=1001, pretrained='imagenet+background')
# Adding 2 channels
first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True)]
first_conv_layer.extend(list(model.features))
model.features= nn.Sequential(*first_conv_layer )
# Bridging for image size

model.last_linear = nn.Linear(1536 , num_classes)
model = nn.DataParallel(model, device_ids=[1, 2, 3, 5, 6, 7])
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

##### Train the model
trainLoss = []
validLoss = []
validAcc = []
total_step = len(train_loader)
for epoch in range(num_epochs):
    exp_lr_scheduler.step()
    meanLoss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        #         print(images.shape)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        meanLoss += loss.cpu().detach().numpy()
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    trainLoss.append(meanLoss / (i + 1))

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        meanLoss = 0
        misclassified = np.zeros(num_classes)
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            meanLoss += loss.cpu().detach().numpy()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i, item in enumerate(predicted != labels):
                if item == 1:
                    misclassified[labels[i]] += 1

        acc = 100 * correct / total
        print('Test Accuracy : {} %, Loss : {:.4f}'.format(100 * correct / total, meanLoss / len(valid_loader)))
        validLoss.append(meanLoss / len(valid_loader))
        validAcc.append(acc)
        misclassified /= (total - correct)
        # print(misclassified*100)
torch.save(model.state_dict(), 'model.ckpt')

x = np.linspace(0,num_epochs,num_epochs)
plt.subplot(1,2,1)
plt.plot(x,trainLoss)
plt.plot(x,validLoss)

plt.subplot(1,2,2)
plt.plot(x,validAcc)
plt.show()
