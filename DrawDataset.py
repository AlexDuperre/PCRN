import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
from tools.renderer import render_depth
import matplotlib.pyplot as plt
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.categories = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl',
                  'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellular telephone', 'chair', 'clock',
                  'computer keyboard', 'dishwasher', 'display', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar',
                  'knife', 'lamp', 'laptop', 'loudspeaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug',
                  'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote control', 'rifle', 'rocket', 'skateboard',
                  'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'trashcan', 'vessel', 'washer']
        self.data_files = []
        for root, dirs, files in tqdm(os.walk(data_dir)):
            for name in files:
                if name.endswith('.png'):
                    self.data_files.append(os.path.join(root, name).replace('\\','/'))

        self.transform = transform

    def __getitem__(self, idx):
        label = self.data_files[idx].replace('\\','/').split('/')[-5]
        label = self.categories.index(label)
        image = Image.open(self.data_files[idx])
        if self.transform:
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.data_files)


class ModelNet40Dataset(Dataset):
    def __init__(self, data_dir, data_type='train', transform = None):
        self.categories = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup',
                  'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
                  'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood',
                  'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.data_files = []
        for root, dirs, files in tqdm(os.walk(data_dir)):
            for name in files:
                if (name.endswith('.png') & (data_type in root)):
                    self.data_files.append(os.path.join(root, name).replace('\\','/'))

        self.transform = transform

    def __getitem__(self, idx):
        label = self.data_files[idx].replace('\\','/').split('/')[-3]
        label = self.categories.index(label)
        image = Image.open(self.data_files[idx])
        if self.transform:
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.data_files)

class MyTransform(object):
    def __call__(self,tensor):
        tensor = torch.abs(tensor-1)
        return tensor[0,:,:].unsqueeze(0)


##############################################################################
#
#                             OFF dataset
#
##############################################################################

class ModelNet40OFFDataset(Dataset):
    def __init__(self, data_dir, data_type='train', transform = None):
        self.categories = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup',
                  'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
                  'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood',
                  'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.data_files = []
        for root, dirs, files in tqdm(os.walk(data_dir)):
            for name in files:
                if (name.endswith('.off') & (data_type in root)):
                    self.data_files.append(os.path.join(root, name).replace('\\','/'))

        self.transform = transform

    def __getitem__(self, idx):
        label = self.data_files[idx].replace('\\','/').split('/')[-3]
        label = self.categories.index(label)
        depth = render_depth(self.data_files[idx])
        if np.max(depth) == 0:
            print(self.data_files[idx])
            depth = render_depth(self.data_files[idx], override=[1,0,0])
        if self.transform:
            image = self.transform(depth)
        return (image, label)

    def __len__(self):
        return len(self.data_files)