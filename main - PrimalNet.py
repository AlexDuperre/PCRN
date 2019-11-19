import os

# Device configuration
DEVICE_ID = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID

import time
import csv
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
from DrawDataset import MyDataset, ModelNet40Dataset, ModelNet40OFFDataset
from DrawDataset import MyTransform
from model2 import PrimalNet
from focalloss import FocalLoss

from sklearn.metrics import confusion_matrix
from Utils import plot_confusion_matrix
###############################################################################################################################
#  Dataset : ShapeNet = 0, ModelNet = 1:
DATASET = 2
pretrained = False


# Hyperparameters
model_degree = 5
possible_monomials  = [3, 9, 19, 34, 55]
monomial_list = possible_monomials[:model_degree-1]

num_epochs = 200
batch_size = 45*DEVICE_ID.split(",").__len__()
ids = range(DEVICE_ID.split(",").__len__())
imsize = 200
print("Using a batch size of :", batch_size)

learning_rate = 0.0001
specific_lr = 0.01
validationRatio = 0.2
validationTestRatio = 0.5


############ Dataset ############
print("Creating Dataset")

if DATASET == 0:
    dataset_name = 'ShapeNet'
    NumClasses = 55
    transformations = transforms.Compose([transforms.Resize((imsize, imsize), interpolation=2),
                                          transforms.ToTensor(),
                                          MyTransform()])
    # ShapeNet
    dataset = MyDataset('/media/SSD/DATA/alex/ShapeNetCoreV2 - Depth/', transform= transformations)
    #dataset = MyDataset('C:/aldupd/RMIT/PCRN/dataset/ShapeNetCoreV2 - Depth', transform= transformations)

    categories = dataset.categories

    # sending to loader
    # torch.manual_seed(0)
    indices = torch.randperm(len(dataset))
    train_indices = indices[:len(indices) - int((validationRatio) * len(dataset))]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)

    valid_train_indices = indices[len(indices) - int(validationRatio * len(dataset)):]
    valid_indices = valid_train_indices[:len(valid_train_indices) - int((validationTestRatio) * len(valid_train_indices))]
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    test_indices =  valid_train_indices[len(valid_train_indices) - int(validationTestRatio * len(valid_train_indices)):]
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

    # Dataset
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               sampler = train_sampler,
                                               shuffle=False,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               sampler = valid_sampler,
                                               shuffle=False,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               sampler = test_sampler,
                                               shuffle=False,
                                               num_workers=0)


if DATASET == 1:
    dataset_name = 'ModelNet40'
    NumClasses = 40

    transformations = transforms.Compose([transforms.Resize((imsize, imsize), interpolation=2),
                                          transforms.ToTensor(),
                                          MyTransform()])
    # transformations = transforms.Compose([transforms.ToTensor()])

    # ModelNet
    trainset = ModelNet40Dataset('/media/SSD/DATA/alex/ModelNet40 - Depth/', transform=transformations)
    #testset = ModelNet40Dataset('/media/SSD/DATA/alex/ModelNet40 - Depth/', data_type='test', transform= transformations)

    #  testset from atomatically rendered .OFF files
    transformations = transforms.Compose([transforms.ToTensor()])
    testset = ModelNet40OFFDataset('/media/SSD/DATA/alex/ModelNet40/', data_type='test', transform=transformations)

    categories = trainset.categories

    # sending to loader
    indices = torch.randperm(len(trainset))
    train_indices = indices[:len(indices) - int((validationRatio) * len(trainset))]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)

    valid_indices = indices[len(indices) - int((validationRatio) * len(trainset)):]
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               shuffle=False,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               shuffle=False,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)

    #valid_loader = test_loader

if DATASET == 2:
    dataset_name = 'ModelNet40OFF'
    NumClasses = 40

    transformations = transforms.Compose([transforms.ToTensor()])

    # ModelNet
    trainset = ModelNet40OFFDataset('/media/SSD/DATA/alex/ModelNet40/', transform=transformations)
    testset = ModelNet40OFFDataset('/media/SSD/DATA/alex/ModelNet40/', data_type='test', transform= transformations)

    categories = trainset.categories

    # sending to loader
    indices = torch.randperm(len(trainset))
    train_indices = indices[:len(indices) - int((validationRatio) * len(trainset))]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)

    valid_indices = indices[len(indices) - int((validationRatio) * len(trainset)):]
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               shuffle=False,
                                               num_workers=3)
    valid_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               shuffle=False,
                                               num_workers=3)
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=3)

    valid_loader = test_loader

# Loading model
if pretrained:
    model = PrimalNet(model_degree, monomial_list,num_classes=55)
    # Load model state dict
    state_dict = torch.load('model.ckpt')

    # fixing the module prefix
    prefix = 'module.'
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in state_dict.items()
                    if k.startswith(prefix)}

    model.load_state_dict(adapted_dict)
    model.last_linear = nn.Linear(2048, 40)
else:
    model = PrimalNet(model_degree, monomial_list, num_classes=NumClasses)

model = nn.DataParallel(model, device_ids=ids)
model = model.cuda()

# Loss and optimizer
#criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 4, 1, 3, 1, 1, 1, 1, 1, 1, 1,
                                                    # 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 3, 1, 3, 1]).float().cuda())
criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss(gamma=2)
optimizer = torch.optim.Adam([{'params':model.module.features[1:-1].parameters()},
                              {'params':model.module.avg_pool.parameters()},
                              {'params':model.module.last_linear.parameters()},
                              {'params':model.module.features[0].parameters(), 'lr':specific_lr}],
                             lr=learning_rate)
# Decay LR by a factor of 0.1 every 7 epochs
step = 150
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.01)

##### Train the model #####
print("Training started")
trainLoss = []
validLoss = []
validAcc = []
best_val_acc = 0
total_step = len(train_loader)
for epoch in range(num_epochs):
    meanLoss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        # Mesh
        xv, yv = torch.meshgrid((torch.linspace(0,1,steps=imsize), torch.linspace(1,0,steps=imsize)))
        xv = xv.cuda()
        yv = yv.cuda()

        batch, channel, height, width = images.shape
        images = [xv.repeat(batch,1,1,1),yv.repeat(batch,1,1,1),images]

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
    # Append mean loss fo graphing and apply lr scheduler
    trainLoss.append(meanLoss / (i + 1))
    exp_lr_scheduler.step()

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        meanLoss = 0
        predictions = np.empty((0, 1))
        ground_truth = np.empty((0, 1))
        for images, labels in valid_loader:
            images = images.cuda()
            labels = labels.cuda()

            # Mesh
            xv, yv = torch.meshgrid((torch.linspace(0, 1, steps=imsize), torch.linspace(1, 0, steps=imsize)))
            xv = xv.cuda()
            yv = yv.cuda()

            batch, channel, height, width = images.shape
            images = [xv.repeat(batch, 1, 1, 1), yv.repeat(batch, 1, 1, 1), images]

            outputs = model(images)
            loss = criterion(outputs, labels)
            meanLoss += loss.cpu().detach().numpy()
            _, predicted = torch.max(outputs.data, 1)
            predictions = np.append(predictions, predicted.cpu().detach().numpy())
            ground_truth = np.append(ground_truth, labels.cpu().detach().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        if acc > best_val_acc:
            best_val_acc = acc

        print('Validation Accuracy : {:.4f} %, Loss : {:.4f}'.format(100 * correct / total, meanLoss / len(valid_loader)))
        validLoss.append(meanLoss / len(valid_loader))
        validAcc.append(acc)


print("Running on testset")
with torch.no_grad():
    correct = 0
    total = 0
    meanLoss = 0
    predictions = np.empty((0, 1))
    ground_truth = np.empty((0, 1))
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()

        #Mesh
        xv, yv = torch.meshgrid((torch.linspace(0, 1, steps=imsize), torch.linspace(1, 0, steps=imsize)))
        xv = xv.cuda()
        yv = yv.cuda()

        batch, channel, height, width = images.shape
        images = [xv.repeat(batch, 1, 1, 1), yv.repeat(batch, 1, 1, 1), images]

        outputs = model(images)
        loss = criterion(outputs, labels)
        meanLoss += loss.cpu().detach().numpy()
        _, predicted = torch.max(outputs.data, 1)
        predictions = np.append(predictions, predicted.cpu().detach().numpy())
        ground_truth = np.append(ground_truth, labels.cpu().detach().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print('Test Accuracy : {} %, Loss : {:.4f}'.format(test_acc, meanLoss / len(test_loader)))



# Print & save results
timestr = time.strftime("%Y%m%d-%H%M%S")
path = "./performances/PrimalNet/" + dataset_name + "/" + timestr
os.makedirs(path)

torch.save(model.state_dict(), path+'/model.ckpt')

x = np.linspace(0,num_epochs,num_epochs)
plt.subplot(1,2,1)
plt.plot(x,trainLoss)
plt.plot(x,validLoss)

plt.subplot(1,2,2)
plt.plot(x,validAcc)
plt.savefig(path+'/learning_curve.png')
plt.show()


# Plotting confusion matrix
cm = confusion_matrix(ground_truth,predictions)
plot_confusion_matrix(cm.astype(np.int64), classes=categories, path=path)

# save specs
dict = {'Dataset' : DATASET,
        'Pretrained' : str(pretrained),
        'Model degree' : model_degree,
        'Num Epochs' : num_epochs,
        'Batch size' : batch_size,
        'Validation ratio' : validationRatio,
        'ValidationTest ratio' : validationTestRatio,
        'Learning rate' : learning_rate,
        'Specific lr' : specific_lr,
        'Device_ID' : DEVICE_ID,
        'imsize' : imsize,
        'Loss fct' : criterion,
        'Lr scheduler step' : step,
        'Best val acc' : best_val_acc,
        'Test acc' : test_acc}
w = csv.writer(open(path+"/specs.csv", "w"))
for key, val in dict.items():
    w.writerow([key, val])
