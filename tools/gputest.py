import os

DEVICE_ID = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID

import torch

a = torch.randn(10)
a.cuda()