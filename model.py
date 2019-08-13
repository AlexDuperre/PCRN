import pretrainedmodels
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from Inception4 import inceptionv4

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #torch.device('cpu')

# Mesh
xv, yv = torch.meshgrid((torch.linspace(0,1,600), torch.linspace(1,0,600)))
xv = xv.reshape(-1).to(device)
yv = yv.reshape(-1).to(device)

class PlaneResLayer(nn.Module):
    def __init__(self):
        super(PlaneResLayer,self).__init__()
        self.plane = Variable(torch.tensor([0.5,0.5,0.5,0.5]), requires_grad=True).to(device)

    def forward(self, image):
        batch, channel, height, width = image.shape
        image = image.reshape(batch,1,-1)
        points = torch.stack([xv.repeat(batch,1,1),yv.repeat(batch,1,1),image]).permute([1,2,3,0])
        res = torch.matmul(points, self.plane[0:3]) + self.plane[3] / torch.norm(self.plane[0:3])
        return res.view(batch,1,height, width).permute(0,1,3,2)


class SphereResLayer(nn.Module):
    def __init__(self):
        super(SphereResLayer,self).__init__()
        self.shpere = Variable(torch.tensor([0.5,0.5,0.5,0.25]),requires_grad=True).to(device)

    def forward(self, image):
        batch, channel, height, width = image.shape
        image = image.reshape(batch,1,-1)
        points = torch.stack([xv.repeat(batch,1,1),yv.repeat(batch,1,1),image]).permute([1,2,3,0])
        res = torch.norm(points-self.shpere[0:3], dim=3) - self.shpere[3]
        return res.view(batch,1,height, width).permute(0,1,3,2)


class CylLayer(nn.Module):
    def __init__(self):
        super(CylLayer,self).__init__()
        self.cylinder = Variable(torch.tensor([0.75, 0.2, 0.1,-0.55,0.2, 0.6, 0.12]), requires_grad=True).to(device)

    def forward(self, image):
        batch, channel, height, width = image.shape
        image = image.reshape(batch,1,-1)
        points = torch.stack([xv.repeat(batch,1,1),yv.repeat(batch,1,1),image]).permute([1,2,3,0])
        AC = points - self.cylinder[0:3]
        AB = self.cylinder[3:6]
        res = torch.norm(torch.cross(AC,AB.repeat(batch,1,height**2,1), dim=3), dim=3) - self.cylinder[6]
        return res.view(batch,1,height, width).permute(0,1,3,2)

# TESTING
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
# Parameters to set
mu_x = 0.5
variance_x = 0.2

mu_y = 0.5
variance_y = 0.2

# Create grid and multivariate normal
x = np.linspace(0, 1, 600)
y = np.linspace(0, 1, 600)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
pdf = rv.pdf(pos)
pdf = torch.tensor(pdf).repeat([2,1,1,1]).float()

# layer = PlaneResLayer()
# res = layer.forward(pdf)


# layer = SphereResLayer()
# res = layer.forward(pdf)

# layer = CylLayer()
# res = layer.forward(pdf)

# print(res.shape)
# plt.imshow(res[0][0].detach().numpy())
# plt.show()
#
# print(res[0][0].numpy()[0,0])
# print(res[0][0].numpy()[0,599])
# print(res[0][0].numpy()[599,599])
# print(res[0][0].numpy()[599,0])

class ResBlock(nn.Module):
    def __init__(self, ResLayer):
        super(ResBlock,self).__init__()
        self.plane_layer_1 = ResLayer()
        self.plane_layer_2 = ResLayer()
        self.plane_layer_3 = ResLayer()

        # Layer 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.relu2 = nn.ReLU()
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)
        self.conv3_drop = nn.Dropout2d()
        self.relu3 = nn.ReLU()
        self.batchNorm3 = nn.BatchNorm2d(1)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.layer1 = nn.Sequential(
            self.conv1,
            self.batchNorm1,
            self.relu1,
            self.maxPool1)

        self.layer2 = nn.Sequential(
            self.conv2,
            self.conv2_drop,
            self.batchNorm2,
            self.relu2)#,
            # self.maxPool2)

        self.layer3 = nn.Sequential(
            self.conv3,
            self.conv3_drop,
            self.batchNorm3,
            self.relu3)#,
            # self.maxPool3)

    def forward(self, image):
        out1 = self.plane_layer_1.forward(image)
        out2 = self.plane_layer_2.forward(image)
        out3 = self.plane_layer_3.forward(image)
        out = torch.cat([out1,out2,out3],dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

# block = ResBlock(PlaneResLayer)
# out = block.forward(pdf)
# plt.imshow(out[0][0].detach().numpy())
# plt.show()

class ResidualNet(nn.Module):
    def __init__(self):
        super(ResidualNet,self).__init__()
        self.block1 = ResBlock(PlaneResLayer)
        self.block2 = ResBlock(SphereResLayer)
        self.block3 = ResBlock(CylLayer)

    def forward(self, image):
        out1 = self.block1.forward(image)
        out2 = self.block2.forward(image)
        out3 = self.block3.forward(image)
        return torch.cat([out1,out2,out3],dim=1)


# model = ResidualNet()
# out = model.forward(pdf)
# plt.imshow(out[0][0].detach().numpy())
# plt.show()



def PCRN(num_classes= 55):
    model = pretrainedmodels.models.resnext101_32x4d(num_classes=1000, pretrained='imagenet')
    residualNet = [ResidualNet()]
    residualNet.extend(list(model.features))
    model.features = nn.Sequential(*residualNet)
    model.last_linear = nn.Linear(32768, num_classes)
    return model

def Resception(num_classes= 55):
    model = inceptionv4(num_classes=1001, pretrained='imagenet+background')
    residualNet = [ResidualNet()]
    residualNet.extend(list(model.features))
    model.features = nn.Sequential(*residualNet)
    model.last_linear = nn.Linear(1536, num_classes)
    return model