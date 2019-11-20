import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from tools.renderer import render_depth
def importOFFfile(filepath):
    # Parse mesh from OFF file
    # filepath = os.fsencode(filepath)
    file = open(filepath, 'r')
    first_line = file.readline().rstrip()

    # handle blank and comment lines after the first line
    # handle OFF characters on the same line as the vcount, fcount and ecount
    # line = file.readline()
    # while line.isspace() or line[0]=='#':
    #    line = file.readline()

    if first_line.split("OFF")[1]:
        line = first_line.split("OFF")[1]
        vcount, fcount, ecount = [int(x) for x in line.split()]
    else:
        line = file.readline()
        vcount, fcount, ecount = [int(x) for x in line.split()]

    verts = []
    X = []
    Y = []
    Z = []
    i = 0;
    while i < vcount:
        line = file.readline()
        if line.isspace():
            continue  # skip empty lines
        try:
            bits = [float(x) for x in line.split()]
            px = bits[0]
            py = bits[1]
            pz = bits[2]

        except ValueError:
            i = i + 1
            continue
        verts.append((px, py, pz))
        X.append(px)
        Y.append(py)
        Z.append(pz)
        i = i + 1

    x = np.asarray(X, dtype=np.float32)
    y = np.asarray(Y, dtype=np.float32)
    z = np.asarray(Z, dtype=np.float32)
    X = (x - min(x)) / (np.max(x) - min(x))
    Y = (y - min(y)) / (np.max(y) - min(y))
    Z = (z - min(z)) / (np.max(z) - min(z))
    point = np.column_stack([[[X, Y, Z]]]).transpose(2,1,0)
    return point


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
        points = importOFFfile(self.data_files[idx])
        if self.transform:
            points = self.transform(points).unsqueeze(0)
        return (points, label)

    def __len__(self):
        return len(self.data_files)


class RandomRot(object):
    def __call__(self,points):
        angle = torch.rand(1)*2*3.1416

        rot_mat = torch.Tensor([[torch.cos(angle), torch.sin(angle), 0],
                               [-torch.sin(angle), torch.cos(angle), 0],
                               [0, 0, 1]])


        points =torch.matmul(points, rot_mat).view(1,3,-1)

        X = (points[0][0] - torch.min(points[0][0])) / (torch.max(points[0][0]) - torch.min(points[0][0]))
        Y = (points[0][1] - torch.min(points[0][1])) / (torch.max(points[0][1]) - torch.min(points[0][1]))
        Z = (points[0][2] - torch.min(points[0][2])) / (torch.max(points[0][2]) - torch.min(points[0][2]))

        points = torch.stack([X, Y, Z], dim=0)
        return points

class DepthGen(object):
    def __call__(selfself, points):
        image = torch.zeros((200, 200))

        # plane properties
        r_o = torch.Tensor([0.5, 0, 0.5])
        n = torch.Tensor([0, 1, 0])
        e_x = torch.Tensor([1, 0, 0])
        e_y = torch.Tensor([0, 0, 1])


        for i, pt in enumerate(points.view(1,-1,3)[0]):
            coordx2d = torch.dot(e_x, pt - r_o)
            coordy2d = torch.dot(e_y, pt - r_o)
            depth = torch.dot(n, pt - r_o)

            pixelx = int(50 + torch.round(100 * (coordx2d + 0.5)))
            pixely = int(150 - torch.round(100 * (coordy2d + 0.5)))

            if (image[pixelx][pixely] > depth) or (image[pixelx][pixely] == 0):
                image[pixely][pixelx] = depth

        return image