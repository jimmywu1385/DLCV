import pickle
import torch.nn as nn
import torchvision.models as models
from transformers import SegformerForSemanticSegmentation

class vgg16FCN32(nn.Module):
    def __init__(self, num_class=7):
        super(vgg16FCN32, self).__init__()
        self.vgg = models.vgg16(weights="DEFAULT").features # B * 16 * 16 * 512
        self.FCN32 = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, num_class, 1, padding="same"),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_class, num_class, 32, stride=32, bias=False)
        )   # B * 512 * 512 * 7

    def forward(self, x):
        x = self.vgg(x)
        x = self.FCN32(x)
        return x

class deeplab3(nn.Module):
    def __init__(self, num_class=7):
        super(deeplab3, self).__init__()
        self.seg = models.segmentation.deeplabv3_resnet101(num_classes=num_class)

    def forward(self, x):
        x = self.seg(x)["out"]
        return x

class segformer(nn.Module):
    def __init__(self, name=None, id2label=None, label2id=None, num_labels=None, config=None):
        super(segformer, self).__init__()
        if config is not None:
            with open(config, "rb") as f:
                self.config = pickle.load(f)
            self.seg = SegformerForSemanticSegmentation(config=self.config)
        else:
            self.seg = SegformerForSemanticSegmentation.from_pretrained(
                            name,
                            num_labels=num_labels,
                            id2label=id2label,
                            label2id=label2id
                        )

    def forward(self, x):
        x = self.seg(**x)
        return x     