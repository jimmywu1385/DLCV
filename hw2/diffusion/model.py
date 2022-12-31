import torch
import torch.nn as nn

class position_embedding(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size    
    
    def forward(self, x):
        frequency = 1.0 / (10000 ** (torch.arange(0, self.hidden_size, 2).float() / self.hidden_size))
        frequency = frequency.to(x.device)
        pos_sin = torch.sin(x.repeat(1, self.hidden_size // 2) * frequency)
        pos_cos = torch.cos(x.repeat(1, self.hidden_size // 2) * frequency)
        pos_embedding = torch.cat([pos_sin, pos_cos], dim=-1)
        return pos_embedding

class basic_block(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None, residual=False):
        super().__init__()
        if mid_channel is None:
            mid_channel = out_channel
        self.residual = residual
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(1, mid_channel),
                    nn.GELU(),
                    nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(1, out_channel),
                )
    
    def forward(self, x):
        if self.residual:
            return nn.functional.gelu(self.conv(x) + x) 
        else:
            return self.conv(x)

class down(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_size=128):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            basic_block(in_channel, in_channel, residual=True),
            basic_block(in_channel, out_channel),
        )

        self.pos_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, out_channel),
        )
    
    def forward(self, x, t):
        x = self.conv(x)
        t = self.pos_embedding(t).view(t.size(0), self.out_channel, 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + t 

class up(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_size=128):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            basic_block(in_channel, in_channel, residual=True),
            basic_block(in_channel, out_channel, in_channel // 2),
        )

        self.pos_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, out_channel),
        )
    
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        t = self.pos_embedding(t).view(t.size(0), self.out_channel, 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + t         

class attention(nn.Module):
    def __init__(self, channel, head=4):
        super().__init__()
        self.channel = channel
        self.m_attention = nn.MultiheadAttention(channel, head, batch_first=True)
        self.norm = nn.LayerNorm([channel])
        self.fc = nn.Sequential(
            nn.LayerNorm([channel]),
            nn.Linear(channel, channel),
            nn.GELU(),
            nn.Linear(channel, channel)            
        )
    
    def forward(self, x):
        channel = x.size(1)
        H = x.size(2)
        W = x.size(3)
        x = x.view(x.size(0), -1, x.size(1))
        atten_x, _ = self.m_attention(self.norm(x), self.norm(x), self.norm(x))
        atten_x = atten_x + x
        atten_x = self.fc(atten_x) +atten_x
        atten_x = atten_x.reshape(x.size(0), channel, H, W)
        return atten_x


class condition_UNet(nn.Module):
    def __init__(self, hidden_size=128, num_class=None):
        super().__init__()
        self.pos_embedding = position_embedding(hidden_size)
        self.hidden_size = hidden_size
        
        if num_class is not None:
            self.label_embedding = nn.Embedding(num_class, hidden_size)

        self.inc = basic_block(3, 64)
        self.down1 = down(64, 128, hidden_size)
        self.sa1 = attention(128)
        self.down2 = down(128, 256, hidden_size)
        self.sa2 = attention(256)
        self.down3 = down(256, 256, hidden_size)
        self.sa3 = attention(256)

        self.bot1 = basic_block(256, 256)
        self.bot2 = basic_block(256, 256)

        self.up1 = up(512, 128, hidden_size)
        self.sa4 = attention(128)
        self.up2 = up(256, 64, hidden_size)
        self.sa5 = attention(64)
        self.up3 = up(128, 64, hidden_size)
        self.sa6 = attention(64)
        self.outc = nn.Conv2d(64, 3, 1)

    def forward(self, x, t, label=None):
        t = t.unsqueeze(-1)
        t = self.pos_embedding(t)
        if label is not None:
            t = self.label_embedding(label) + t

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)

        return output




