import torch
from torch.utils.data import Dataset
from PIL import Image
import csv
import os

class uda_data(Dataset):
    def __init__(self, data_root, csv_path=None, transform=None):
        self.root = data_root
        self.test = False
        if csv_path is None:
            self.csv_path = csv_path
            self.test = True
        if self.test:
            self.filenames = [i for i in os.listdir(data_root)]
        else:
            self.filenames = []
            self.label = []
            with open(csv_path, newline='') as csvfile:
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
        if self.transform is not None:
            image = self.transform(image)
        if self.test:
            return file, image
        else:
            label = torch.tensor(self.label[index])
            return label, image