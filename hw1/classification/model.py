import torch.nn as nn
import torchvision.models as models

class resnet(nn.Module):
    def __init__(self, test=False):
        super(resnet, self).__init__()
        if test:
            self.model = models.resnext101_32x8d()
        else: 
            self.model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(2048, 50)

    def forward(self, x):
        x = self.model(x)
        return x

class my_cnn(nn.Module):
    def __init__(self):
        super(my_cnn, self).__init__()
        self.in_channel = 64

        self.conv1 = conv1(3)
        self.conv2 = self._make_layer(bottleneck, 64, 3, 1)
        self.conv3 = self._make_layer(bottleneck, 128, 4, 2)
        self.conv4 = self._make_layer(bottleneck, 256, 23, 2)
        self.conv5 = self._make_layer(bottleneck, 512, 3, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 50)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output 

    def _make_layer(self, block, out_channel, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layer = []
        for stride in strides:
            layer.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel * 4
        return nn.Sequential(*layer)

class conv1(nn.Module):
    def __init__(self, in_channel):
        super(conv1, self).__init__()
        self.conv = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nn.ReLU(inplace=True)(x)
        return x

class bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(bottleneck, self).__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, stride=stride, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel*4, kernel_size=1),
            nn.BatchNorm2d(out_channel*4),
        )

        self.shorcut = nn.Sequential()

        if stride != 1 or in_channel != out_channel*4:
            self.shorcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel*4, stride=stride, kernel_size=1),
                nn.BatchNorm2d(out_channel*4),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_block(x) + self.shorcut(x))