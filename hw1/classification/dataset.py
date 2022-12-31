import os

from torch.utils.data import Dataset
from PIL import Image

class Cls_data(Dataset):
    def __init__(self, data_root, transform=None, test=False):
        self.root = data_root
        self.filenames = [i for i in os.listdir(data_root)]
        self.transform = transform
        self.test = test

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
            label = int(file.split("_")[0])
            return label, image
