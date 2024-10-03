import torch
from torch import nn

class Bn_Cov2d_Relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bn_Cov2d_Relu, self).__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.seq(x)
class channel_attention(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(channel_attention, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, padding=0)
        self.Bn_Cov2d_Relu1 = Bn_Cov2d_Relu(64, 64)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 64 // ratio),
            nn.ReLU(),
            nn.Linear(64 // ratio, 64),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Bn_Cov2d_Relu1(x)
        b,c,h,w = x.size()
        max_pool_out = self.max_pool(x).view([b,c])
        avg_pool_out = self.avg_pool(x).view([b,c])
        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b,c,1,1])
        return out*x
if __name__ == '__main__':
    model = channel_attention(in_channels=1, ratio=4)
    x = torch.randn((1,1,640,480))
    y = model(x)
    print(y.shape)