import torch
import torch.nn as nn

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 5),
            nn.BatchNorm2d(128),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Flatten(),
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
        )
    
    def forward(self, x, alpha):
        x = x.expand(x.size(0), 3, 28, 28)
        feature = self.feature(x)
        reverse_feature = ReverseLayerF.apply(feature, alpha)

        class_out = self.class_classifier(feature)
        domain_out = self.domain_classifier(reverse_feature)
        return class_out, domain_out
