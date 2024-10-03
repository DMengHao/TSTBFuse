import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False,stride=1):
        super().__init__()
        self.ReLu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=7, padding=3)
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
        y = self.ReLu(y)
        if self.conv3:
            x= self.conv3(x)
        out = self.ReLu(x+y)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.s1 = nn.Sequential(
            Residual(64, 128, use_1conv=True, stride=1),
            Residual(128, 128, use_1conv=False, stride=1),
        )
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.s1(x)
        x = self.conv2(x)
        return x

if __name__ == '__main__':
    x = torch.rand(1, 1, 640, 480)
    model = ResNet18()
    y = model(x)
    print(y.shape)