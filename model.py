import pretrainedmodels
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from Inception4 import inceptionv4

# Device configuration
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') #torch.device('cpu')

class PlaneResLayer(nn.Module):
    def __init__(self):
        super(PlaneResLayer,self).__init__()
        self.plane = Variable(torch.tensor([0.5,0.5,0.5,0.5], device="cuda"), requires_grad=True)
	# Mesh
        xv, yv = torch.meshgrid((torch.linspace(0,1,steps=600, device="cuda"), torch.linspace(1,0,steps=600, device="cuda")))
        self.xv = xv.reshape(-1)
        self.yv = yv.reshape(-1)


    def forward(self, image):
        batch, channel, height, width = image.shape
        image = image.reshape(batch,1,-1)
        points = torch.stack([self.xv.repeat(batch,1,1),self.yv.repeat(batch,1,1),image]).permute([1,2,3,0])
        res = torch.matmul(points, self.plane[0:3]) + self.plane[3] / torch.norm(self.plane[0:3])
        return res.view(batch,1,height, width).permute(0,1,3,2)


class SphereResLayer(nn.Module):
    def __init__(self):
        super(SphereResLayer,self).__init__()
        self.shpere = Variable(torch.tensor([0.5,0.5,0.5,0.25], device="cuda"),requires_grad=True)
	# Mesh
        xv, yv = torch.meshgrid((torch.linspace(0,1,steps=600, device="cuda"), torch.linspace(1,0,steps=600, device="cuda")))
        self.xv = xv.reshape(-1)
        self.yv = yv.reshape(-1)

    def forward(self, image):
        batch, channel, height, width = image.shape
        image = image.reshape(batch,1,-1)
        points = torch.stack([self.xv.repeat(batch,1,1),self.yv.repeat(batch,1,1),image]).permute([1,2,3,0])
        res = torch.norm(points-self.shpere[0:3], dim=3) - self.shpere[3]
        return res.view(batch,1,height, width).permute(0,1,3,2)



class CylResLayer(nn.Module):
    def __init__(self,batch, width, height):
        super(CylResLayer,self).__init__()
        self.cylinder = Parameter(torch.tensor([0.75, 0.2, 0.1,-0.55,0.2, 0.6, 0.12]), requires_grad=True)
        self.batch = batch
        self.width = width
        self.height = height


    def forward(self, points):
        batch, channel, height, width = points.shape
        #AC = points - self.cylinder[0:3]
        #AB = self.cylinder[3:6]
        n = self.cylinder[3:6] / (torch.norm(self.cylinder[3:6]) + 0.0001)
        AD = torch.abs(torch.matmul(points - self.cylinder[0:3], n.transpose(0, -1)))
        AC = torch.norm(points - self.cylinder[0:3], dim=3)
        res = torch.sqrt(AC ** 2. - AD ** 2.) - self.cylinder[6]
        return res.view(batch,1,self.height, self.width).permute(0,1,3,2)

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
        self.res_layer_1 = ResLayer
        self.res_layer_2 = ResLayer
        self.res_layer_3 = ResLayer

        # Layer 2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.relu2 = nn.ReLU()
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2)
        self.conv3_drop = nn.Dropout2d()
        self.relu3 = nn.ReLU()
        self.batchNorm3 = nn.BatchNorm2d(16)
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
        out1 = self.res_layer_1(image)
        out2 = self.res_layer_2(image)
        out3 = self.res_layer_3(image)
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
        self.block1 = ResBlock(PlaneResLayer(batch, width, height))
        self.block2 = ResBlock(SphereResLayer(batch, width, height))
        self.block3 = ResBlock(CylResLayer(batch, width, height))

    def forward(self, image):
        out1 = self.block1(image)
        out2 = self.block2(image)
        out3 = self.block3(image)
        return torch.cat([out1,out2,out3],dim=1)


# model = ResidualNet()
# out = model.forward(pdf)
# plt.imshow(out[0][0].detach().numpy())
# plt.show()



def PCRN(num_classes= 55):
    model = pretrainedmodels.models.resnext101_32x4d(num_classes=1000, pretrained=None)
    # Change first conv from Resnext (Removes bottlneck of only 3 channles)
    modelFeatures = list(model.features)
    modelFeatures.pop(0)
    first_conv_layer = [nn.Conv2d(48, 64, kernel_size=7, stride=2, padding=3, dilation=1, groups=1, bias=False)]
    first_conv_layer.extend(modelFeatures)
    model.features = nn.Sequential(*first_conv_layer)

    # Add Residual Net module
    residualNet = [ResidualNet(batch, width, height)]
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
