import torch
from torch import nn
from torch.nn.modules.activation import ReLU

class channel_attention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b,c,h,w = x.size()
        max_pool_out = self.max_pool(x).view([b,c])
        avg_pool_out = self.avg_pool(x).view([b,c])
        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b,c,1,1])
        return out*x

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_pool_out,_ = torch.max(x,dim=1,keepdim=True)
        mean_pool_out = torch.mean(x,dim=1,keepdim=True)
        pol_out = torch.cat([max_pool_out,mean_pool_out],dim=1)
        out = self.conv1(pol_out)
        out = self.sigmoid(out)
        return out * x

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = channel_attention(in_channels, ratio)
        self.spatial_attention = spatial_attention()
    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out

if __name__ == '__main__':
    model = CBAM(in_channels=64, ratio=16)
    inputs = torch.randn((4, 64, 128, 128))
    out = model(inputs)
    print(out.shape)
