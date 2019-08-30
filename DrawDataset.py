from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
categories = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellular telephone', 'chair', 'clock', 'computer keyboard', 'dishwasher', 'display', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'loudspeaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote control', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'trashcan', 'vessel', 'washer']


class MyDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_files = []
        for root, dirs, files in tqdm(os.walk(data_dir)):
            for name in files:
                if name.endswith('.png'):
                    self.data_files.append(os.path.join(root, name).replace('\\','/'))
        self.transform = transform

    def __getitem__(self, idx):
        label = self.data_files[idx].replace('\\','/').split('/')[-5]
        label = categories.index(label)
        image = Image.open(self.data_files[idx])
        if self.transform:
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.data_files)
