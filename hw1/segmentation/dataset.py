import os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

def read_masks(labels):
    labels = np.array(labels)
    rgb_mask = (labels >= 128).astype(int)
    rgb_mask = 4 * rgb_mask[:, :, 0] + 2 * rgb_mask[:, :, 1] + rgb_mask[:, :, 2]
    y = np.empty(rgb_mask.shape,dtype=int)
    y[rgb_mask == 3] = 0 
    y[rgb_mask == 6] = 1
    y[rgb_mask == 5] = 2
    y[rgb_mask == 2] = 3
    y[rgb_mask == 1] = 4
    y[rgb_mask == 7] = 5 
    y[rgb_mask == 0] = 6
    y[rgb_mask == 4] = 6
    return y

class Seg_data(Dataset):
    def __init__(self, data_root, transform=None, test=False):
        self.root = data_root
        self.test = test
        self.sat = sorted([i for i in os.listdir(data_root) if i.endswith(".jpg")])
        if not self.test:
            self.mask = sorted([i for i in os.listdir(data_root) if i.endswith(".png")])
        self.transform = transform
        

    def __len__(self):
        return len(self.sat)

    def __getitem__(self, index):
        filename = self.sat[index]
        sat = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            sat = self.transform(sat)
        if self.test:
            return filename, sat
        else:
            mask = Image.open(os.path.join(self.root, self.mask[index]))
            mask = read_masks(mask)
            return torch.LongTensor(mask), sat


class Segf_data(Dataset):
    def __init__(self, data_root, transform, test=False):
        self.root = data_root
        self.test = test
        self.sat = sorted([i for i in os.listdir(data_root) if i.endswith(".jpg")])
        if not self.test:
            self.mask = sorted([i for i in os.listdir(data_root) if i.endswith(".png")])
        self.transform = transform

    def __len__(self):
        return len(self.sat)

    def __getitem__(self, index):
        filename = self.sat[index]
        sat = Image.open(os.path.join(self.root, filename))
        if self.test:
            return filename, sat
        else:
            mask = Image.open(os.path.join(self.root, self.mask[index]))
            mask = read_masks(mask)
            return mask, sat

    def collate_fn(self, sample):
        if self.test:
            filename = [x[0] for x in sample]
            images = [x[1] for x in sample]
            inputs = self.transform(images, return_tensors="pt")
            return filename, inputs
        else:
            labels = [x[0] for x in sample]
            images = [x[1] for x in sample]
            inputs = self.transform(images, labels, return_tensors="pt")
            return inputs