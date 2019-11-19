import pretrainedmodels
import numpy as np
import torch
import torch.nn as nn
import itertools
from functools import reduce


def get_polynomial(vars, power):
    List = []
    for L in range(1, power + 1):
        for subset in itertools.combinations_with_replacement(vars, L):
            List.append(reduce(lambda x, y: x * y, list(subset)))

    return torch.cat(List, dim=1)


def get_Input(vars, degree):
    List = []
    for i in range(1, degree+1):
        List.append(get_polynomial(vars, i))

    return tuple(List)

class SplitCNN(nn.Module):
    def __init__(self,channels=3,kernel=5,split=[3,1]):
        super(SplitCNN, self).__init__()

        self.channels = channels
        # self.depth = depth
        self.kernel = kernel
        self.split = split
        # self.channelsAux = 0

        self.dcnn = nn.ModuleList()
        for i in range(len(split)):
            self.dcnn.append(nn.utils.weight_norm(nn.Conv2d(
                                in_channels=split[i],
                                out_channels=self.channels*split[i],
                                kernel_size=self.kernel,
                                stride=1,
                                padding=0,
                                groups=split[i]),
                                name='weight'))

    def forward(self, x):
        out = []
        for j in range(len(self.split)):
            split_conv_out = self.dcnn[j](x[j])
            b, c, h, w = split_conv_out.shape
            out.append(split_conv_out.reshape(b, c//3, 3, h, w).sum(1))

        return torch.cat(out,1)


class PolyLayer(nn.Module):
    def __init__(self,degree, split):
        super(PolyLayer,self).__init__()
        self.degree = degree
        self.split = split
        self.conv = SplitCNN(split= self.split,kernel=1)
        self.batch_norm = nn.BatchNorm2d(3*len(split))
        self.activ = nn.Tanh()


    def forward(self, channels):
        input = get_Input(channels, self.degree)
        out = self.conv(input)
        out = self.batch_norm(out)
        out = self.activ(out)
        return out

def PrimalNet(degree, split, num_classes= 55):
    model = pretrainedmodels.models.resnext101_32x4d(num_classes=1000, pretrained=None)
    # Change first conv from Resnext
    modelFeatures = list(model.features)
    modelFeatures.pop(0)
    first_conv_layer = [nn.Conv2d(3*len(split), 64, kernel_size=7, stride=2, padding=3, dilation=1, groups=1, bias=False)]
    first_conv_layer.extend(modelFeatures)
    model.features = nn.Sequential(*first_conv_layer)

    # Add Polyconv module before
    PolyConv = [PolyLayer(degree, split)]
    PolyConv.extend(list(model.features))
    model.features = nn.Sequential(*PolyConv)
    model.last_linear = nn.Linear(2048, num_classes)
    return model
