import torch.nn as nn
import torchvision.models as models

class resnet(nn.Module):
    def __init__(self, num_class, backbone=None, freeze=False):
        super(resnet, self).__init__()
        if backbone == None:
            self.backbone = models.resnet50(weights=None)
        else:
            self.backbone = backbone

        if freeze:
            for name, child in self.backbone.named_children():
                for param in child.parameters():
                    param.requires_grad = False

        self.feature = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_class)
        )
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), 2048)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = resnet(10)
    print(model)