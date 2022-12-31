import os
import csv

from torch.utils.data import Dataset
from PIL import Image

class Cls_data(Dataset):
    def __init__(self, data_root, label_path, transform=None, label_dic=None):
        self.root = data_root
        self.transform = transform
        self.label_path = label_path

        self.id = []
        self.filenames = []
        self.label = []
        with open(label_path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            next(rows, None)
            for i in rows:
                self.id.append(i[0])
                self.filenames.append(i[1])
                self.label.append(i[2])

        if label_dic == None:
            label_set = set(self.label)
            label_set = sorted(label_set)
            self.label_dic = {label: i for i, label in enumerate(label_set)}
        else:
            self.label_dic = label_dic

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        id = self.id[index]
        file = self.filenames[index]
        label = self.label[index]
        label = self.label_dic[label]
        image = Image.open(os.path.join(self.root, file)).convert("RGB") 
        if self.transform is not None:
            image = self.transform(image)
        return id, file, image, label

    @property
    def get_label_dic(self):
        return self.label_dic