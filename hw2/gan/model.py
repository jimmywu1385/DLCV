import torch.nn as nn

class b_Discriminator(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(b_Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channel, hidden_channel*2, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channel*2, hidden_channel*4, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channel*4, hidden_channel*8, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channel*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class b_Generator(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(b_Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channel, hidden_channel*8, 4, 1, 0, bias=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_channel*8, hidden_channel*4, 4, 2, 1, bias=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_channel*4, hidden_channel*2, 4, 2, 1, bias=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_channel*2, hidden_channel, 4, 2, 1, bias=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_channel, out_channel, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_channel, hidden_channel*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channel*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_channel*2, hidden_channel*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channel*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_channel*4, hidden_channel*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channel*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_channel*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channel, hidden_channel*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_channel*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_channel*8, hidden_channel*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channel*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_channel*4, hidden_channel*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channel*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_channel*2, hidden_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_channel, out_channel, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, x):
        return self.model(x)
