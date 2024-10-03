import torch
from torch import nn

first_output_channel = 14
input_channels = 2
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

class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.Bn_Cov2d_Relu1 = Bn_Cov2d_Relu(input_channels, out_channels=first_output_channel)
        self.Bn_Cov2d_Relu2 = Bn_Cov2d_Relu(16, out_channels=16)
        self.Bn_Cov2d_Relu3 = Bn_Cov2d_Relu(32, out_channels=16)
        self.Bn_Cov2d_Relu4 = Bn_Cov2d_Relu(48, out_channels=64)


    def forward(self, x1,x2):
        x = torch.cat((x1,x2),dim=1)
        x1 = x
        x2 = self.Bn_Cov2d_Relu1(x)
        y1 = torch.cat([x1, x2], dim=1)
        x3 = self.Bn_Cov2d_Relu2(y1)
        y2 = torch.cat([x1,x2,x3], dim=1)
        x4 = self.Bn_Cov2d_Relu3(y2)
        y3 = torch.cat([x1,x2,x3,x4], dim=1)
        x5 = self.Bn_Cov2d_Relu4(y3)
        return x5

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, stride=1):
        super().__init__()
        self.ReLu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1)
        else:
            self.conv3 = None
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.ReLu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.conv3:
            x= self.conv3(x)
        out = self.ReLu(x+y)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.s1 = nn.Sequential(
            Residual(64, 128, use_1conv=True, stride=2),
            Residual(128, 128, use_1conv=False, stride=2),
        )
    def forward(self, x):
        x = self.s1(x)
        return x

class Dense_Resnet_M(nn.Module):
    def __init__(self):
        super(Dense_Resnet_M, self).__init__()
        self.Denseblock =  DenseBlock(2)
        self.resnet = ResNet18()
        # self.reduce = nn.Conv2d(128, 64, kernel_size=1)
    def forward(self, x1, x2):
        x = torch.cat([x1,x2], dim=1)
        x = self.Denseblock(x)
        x = self.resnet(x)
        # x = self.reduce(x)
        return x
if __name__ == '__main__':
    model = DenseBlock()
    x1 = torch.randn(1, 1, 640, 480)
    x2 = torch.randn(1, 1, 640, 480)
    y = model(x1,x2)
    print(y.shape)