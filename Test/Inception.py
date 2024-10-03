import torch
from torch import nn
def mean_normalization_tensor(data):
    mean = torch.mean(data, dim=(1, 2, 3), keepdim=True)  # 计算均值
    std = torch.std(data, dim=(1, 2, 3), keepdim=True)    # 计算标准差
    normalized_data = (data - mean) / std  # 均值归一化
    return normalized_data
class Spatial_attention(nn.Module):
    def __init__(self,  channel, reduction=4):
        super(Spatial_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel),
            nn.Sigmoid()
        )
    def forward(self, input):
        b,c,h,w = input.shape
        avg = self.avg_pool(input).view([b,c])
        fc = self.fc(avg).view(b,c,1,1)
        return input*fc
class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()

        # 路线1，单1×1卷积层
        self.p1_1 = nn.Conv2d(in_channels=64, out_channels=63, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(63)
        self.ReLU1 = nn.ReLU()
        # 路线2，1×1卷积层, 3×3的卷积
        self.p2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.ReLU2 = nn.ReLU()
        self.spatial_attention = Spatial_attention(64)

        # 路线3，1×1卷积层, 7*7的卷积
        self.p3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.ReLU3 = nn.ReLU()
        # 路线4，3×3的最大池化, 1×1的卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        # self.ReLU = nn.LeakyReLU()


    def forward(self, x):
        t = x-0.5
        x = x-0.5
        x = self.conv1(x)
        # x = self.bn_conv1(x)
        x = mean_normalization_tensor(x)
        x = self.ReLU(x)

        # p1 = self.ReLU(self.bn1(self.p1_1(x)))
        p1 = self.p1_1(x)
        p1 = mean_normalization_tensor(p1)
        p1 = self.ReLU(p1)

        # p2 = self.ReLU(self.bn2(self.p2_1(x)))
        p2 = self.p2_1(x)
        p2 = mean_normalization_tensor(p2)
        p2 = self.ReLU(p2)
        p2 = self.spatial_attention(p2)

        # p3 = self.ReLU(self.bn3(self.p3_1(x)))
        p3 = self.p3_1(x)
        p3 = mean_normalization_tensor(p3)
        p3 = self.ReLU(p3)
        p3 = self.spatial_attention(p3)

        # p4 = self.ReLU(self.bn4(self.p4_2(self.p4_1(x))))
        p4 = self.p4_1(x)
        p4 = self.p4_2(x)
        p4 = mean_normalization_tensor(p4)
        p4 = self.ReLU(p4)
        p4 = self.spatial_attention(p4)



        out = torch.cat((t, p1, p2, p3, p4), dim=1)
        out = self.conv2(out)
        # out = self.bn2(out)
        out = mean_normalization_tensor(out)
        out = self.ReLU(out)
        return out
if __name__ == '__main__':
    # a = torch.randn(1,64,640,480)
    # m = Spatial_attention(64)
    # n = m(a)
    # print(n.shape)
    x = torch.randn(1, 1, 640, 480)
    model = Inception()
    y = model(x)
    print(y.shape)
