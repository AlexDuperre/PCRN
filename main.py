import os

# Device configuration
DEVICE_ID = "5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from DrawDataset import MyDataset
from DrawDataset import MyTransform
from model import PCRN
from model import WeightClipper
from sklearn.metrics import confusion_matrix
from Utils import plot_confusion_matrix
#from torch.utils.tensorboard import SummaryWriter

###############################################################################################################################
# Initialize Tensorboard Summaries
#writer = SummaryWriter()



# Hyper parameters
num_epochs = 10
num_classes = 55
batch_size = 8*DEVICE_ID.split(",").__len__()

print("Using a batch size of :", batch_size)

learning_rate = 0.0001
validationRatio = 0.3
validationTestRatio = 0.5

# train data
transformations = transforms.Compose([transforms.ToTensor(),
                                      MyTransform()])
print("Creating Dataset")
dataset = MyDataset('/media/SSD/DATA/alex/ShapeNetCoreV2 - Depth/', transform= transformations)

#dataset = MyDataset('C:/aldupd/RMIT/PCRN/dataset/ShapeNetCoreV2 - Depth', transform= transformations)

# sending to loader
torch.manual_seed(0)
indices = torch.randperm(len(dataset))
train_indices = indices[:len(indices) - int((validationRatio) * len(dataset))]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)

valid_train_indices = indices[len(indices) - int(validationRatio * len(dataset)):]
valid_indices = valid_train_indices[:len(valid_train_indices) - int((validationTestRatio) * len(valid_train_indices))]
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

test_indices = valid_train_indices[len(valid_train_indices) - int(validationTestRatio * len(valid_train_indices)):]
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

# Dataset
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           sampler = train_sampler,
                                           shuffle=False,
                                           num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size*4,
                                           sampler = valid_sampler,
                                           shuffle=False,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size*4,
                                           sampler = test_sampler,
                                           shuffle=False,
                                           num_workers=0)


# Loading model
model = PCRN(batch_size,600,600) #.to(device)
model = nn.DataParallel(model,  device_ids=[0,1,2])
model = model.cuda()
clipper = WeightClipper()

# Loss and optimizer and adapted lr for ResidualNet
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params':model.module.features[0]._parameters, 'lr':0.1}], lr=learning_rate)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

##### Train the model #####
print("Training started")

f = open("planes.txt", "w+")
g = open("spheres.txt", "w+")
h = open("cyl.txt", "w+")

trainLoss = []
validLoss = []
validAcc = []
total_step = len(train_loader)
for epoch in range(num_epochs):
    meanLoss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        # Mesh
        xv, yv = torch.meshgrid((torch.linspace(0,1,steps=600), torch.linspace(1,0,steps=600)))
        xv = xv.reshape(-1).cuda()
        yv = yv.reshape(-1).cuda()

        batch, channel, height, width = images.shape
        images = images.reshape(batch,1,-1)
        images = torch.stack([xv.repeat(batch,1,1),yv.repeat(batch,1,1),images]).permute([1,2,3,0]).cuda()

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)
        meanLoss += loss.cpu().detach().numpy()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Restricts shapes into the unit cube
        model.apply(clipper)

        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            # print shape parameters to a text file
            f.write(str(list(model.module.features[0].block1.res_layer_1.plane.detach().cpu().numpy())) + "\n")
            g.write(str(list(model.module.features[0].block2.res_layer_1.sphere.detach().cpu().numpy())) + "\n")
            h.write(str(list(model.module.features[0].block3.res_layer_1.cylinder.detach().cpu().numpy())) + "\n")

    # Append mean loss fo graphing and apply lr scheduler
    trainLoss.append(meanLoss / (i + 1))
    exp_lr_scheduler.step()

    # Validation of the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        meanLoss = 0
        misclassified = np.zeros(num_classes)
        for images, labels in valid_loader:
            images = images.cuda()
            labels = labels.cuda()

            # Mesh
            xv, yv = torch.meshgrid((torch.linspace(0, 1, steps=600), torch.linspace(1, 0, steps=600)))
            xv = xv.reshape(-1).cuda()
            yv = yv.reshape(-1).cuda()

            batch, channel, height, width = images.shape
            images = images.reshape(batch, 1, -1)
            images = torch.stack([xv.repeat(batch, 1, 1), yv.repeat(batch, 1, 1), images]).permute([1, 2, 3, 0]).cuda()

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
        print('Validation Accuracy : {} %, Loss : {:.4f}'.format(100 * correct / total, meanLoss / len(valid_loader)))
        validLoss.append(meanLoss / len(valid_loader))
        validAcc.append(acc)
        misclassified /= (total - correct)
        # print(misclassified*100)

print("Running on test set")
with torch.no_grad():
    correct = 0
    total = 0
    meanLoss = 0
    predictions = np.empty((0,1))
    ground_truth = np.empty((0,1))
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()

        # Mesh
        xv, yv = torch.meshgrid((torch.linspace(0, 1, steps=600), torch.linspace(1, 0, steps=600)))
        xv = xv.reshape(-1).cuda()
        yv = yv.reshape(-1).cuda()

        batch, channel, height, width = images.shape
        images = images.reshape(batch, 1, -1)
        images = torch.stack([xv.repeat(batch, 1, 1), yv.repeat(batch, 1, 1), images]).permute([1, 2, 3, 0]).cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)
        meanLoss += loss.cpu().detach().numpy()
        _, predicted = torch.max(outputs.data, 1)
        predictions = np.append(predictions,predicted.cpu().detach().numpy())
        ground_truth = np.append(ground_truth,labels.cpu().detach().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if count == 20:
            break
        count += 1
    print('Test Accuracy : {} %, Loss : {:.4f}'.format(100 * correct / total, meanLoss / len(test_loader)))


# Close files
f.close()
g.close()
h.close()

# torch.save(model.state_dict(), 'model.ckpt')
#
# x = np.linspace(0,num_epochs,num_epochs)
# plt.subplot(1,2,1)
# plt.plot(x,trainLoss)
# plt.plot(x,validLoss)
#
# plt.subplot(1,2,2)
# plt.plot(x,validAcc)
# plt.show()

# Plotting confusion matrix
categories = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellular telephone', 'chair', 'clock', 'computer keyboard', 'dishwasher', 'display', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'loudspeaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote control', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'trashcan', 'vessel', 'washer']

cm = confusion_matrix(ground_truth,predictions)
plot_confusion_matrix(cm.astype(np.int64), classes=categories)

# Plotting confusion matrix
categories = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellular telephone', 'chair', 'clock', 'computer keyboard', 'dishwasher', 'display', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'loudspeaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote control', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'trashcan', 'vessel', 'washer']

cm = confusion_matrix(ground_truth,predictions)
plot_confusion_matrix(cm.astype(np.int64), classes=categories)