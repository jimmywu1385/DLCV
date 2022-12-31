import os
import json

from torch.utils.data import Dataset
from PIL import Image
import torch

class caption_data(Dataset):
    def __init__(self, data_root, tokenizer, max_len, label_path=None, transform=None):
        self.root = data_root
        self.test = False
        self.tokenizer = tokenizer
        self.max_len = max_len
        if label_path == None:
            self.test = True

        if self.test:
            self.filenames = [i for i in os.listdir(data_root)]
        else:
            self.label_path = label_path
            with open(label_path) as f:
                self.labels = json.load(f)
            image = {}
            for i in self.labels["images"]:
                image[i["id"]] = i["file_name"]

            self.filenames = []
            self.captions = []
            for i in self.labels["annotations"]:
                self.filenames.append(image[i["image_id"]])
                self.captions.append(i["caption"])

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        file = self.filenames[index]
        image = Image.open(os.path.join(self.root, file)).convert("RGB") 
        if self.transform is not None:
            image = self.transform(image)
        if self.test:
            return file, image
        else:
            caption = self.captions[index]
            caption = self.tokenize(caption)
            return caption, image

    def get_key_padding_mask(self, tokens):
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 0] = -torch.inf
        return key_padding_mask

    def tokenize(self, caption):
        caption = self.tokenizer.encode(caption)
        caption = caption.ids + [0] * (self.max_len-len(caption))
        caption_in = torch.tensor(caption[:-1])
        caption_out = torch.tensor(caption[1:])
        return {"caption_in" : caption_in,
                "caption_out" : caption_out,
                "mask" : self.get_key_padding_mask(caption_in),
        }

