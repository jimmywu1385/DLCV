import os

from torch.utils.data import Dataset
from PIL import Image

class CLIP_data(Dataset):
    def __init__(self, data_root, transform=None):
        self.root = data_root
        self.filenames = [i for i in os.listdir(data_root)]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        file = self.filenames[index]
        image = Image.open(os.path.join(self.root, file)).convert("RGB") 
        if self.transform is not None:
            image = self.transform(image)
        return file, image