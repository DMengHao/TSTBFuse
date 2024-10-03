import torch
from torch import nn
kernel_size=7
def mean_normalization_tensor(data):
    mean = torch.mean(data, dim=(1, 2, 3), keepdim=True)  # 计算均值
    std = torch.std(data, dim=(1, 2, 3), keepdim=True)    # 计算标准差
    normalized_data = (data - mean) / std  # 均值归一化
    return normalized_data
class channel_attention(nn.Module):
    def __init__(self, in_channels, ratio=4):
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
    def __init__(self):
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

class VIFEM(nn.Module):
    def __init__(self):
        super(VIFEM, self).__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True)
        )
        self.ReLU = nn.ReLU(True)
        # self.ReLU = nn.LeakyReLU(negative_slope=0.5, inplace=True)

        self.sam = spatial_attention()
        self.cam = channel_attention(32)

    def forward(self,x):
        x = x-0.5
        x = self.s1(x)
        x =  mean_normalization_tensor(x)
        x = self.ReLU(x)
        sam = self.sam(x)
        cam = self.cam(x)
        return torch.cat((sam,cam),1)
if __name__ == '__main__':
    x = torch.randn(1, 1, 640, 480)
    model = VIFEM()
    y = model.forward(x)
    print(y.shape)
