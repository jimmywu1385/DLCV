import torch
from torch.utils.data import Dataset
from PIL import Image
import csv
import os

class dif_data(Dataset):
    def __init__(self, data_root, label_path, transform=None):
        self.root = data_root
        self.label_path = label_path
        self.filenames = []
        self.label = []
        with open(label_path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            next(rows, None)
            for i in rows:
                self.filenames.append(i[0])
                self.label.append(int(i[1]))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        file = self.filenames[index]
        image = Image.open(os.path.join(self.root, file))
        label = torch.tensor(self.label[index])
        if self.transform is not None:
            image = self.transform(image)
        return label, image