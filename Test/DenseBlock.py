import torch
from torch import nn

def mean_normalization_tensor(data):
    mean = torch.mean(data, dim=(1, 2, 3), keepdim=True)  # 计算均值
    std = torch.std(data, dim=(1, 2, 3), keepdim=True)    # 计算标准差
    normalized_data = (data - mean) / std  # 均值归一化
    return normalized_data
first_output_channel = 62
class Bn_Cov2d_Relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bn_Cov2d_Relu, self).__init__()
        self.seq = nn.Sequential(
            # nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )
        self.ReLU = nn.ReLU(inplace=True)
        self.LeakRelu = nn.LeakyReLU(negative_slope=0.5, inplace=True)
    def forward(self, x):
        x = self.seq(x)
        x = mean_normalization_tensor(x)
        x = self.ReLU(x)
        # x = self.LeakRelu(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.Bn_Cov2d_Relu1 = Bn_Cov2d_Relu(2, out_channels=first_output_channel)
        self.Bn_Cov2d_Relu2 = Bn_Cov2d_Relu(64, out_channels=64)
        self.Bn_Cov2d_Relu3 = Bn_Cov2d_Relu(128, out_channels=64)
        self.Bn_Cov2d_Relu4 = Bn_Cov2d_Relu(192, out_channels=64)
        self.Bn_Cov2d_Relu5 = Bn_Cov2d_Relu(256, out_channels=64)
        # self.Conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0)
        # self.ReLU = nn.ReLU(inplace=True)


    def forward(self, x1,x2):
        x1 = x1-0.5
        x2 = x2-0.5
        # x3 = x3-0.5
        x = torch.cat([x1, x2], dim=1)
        x1 = x
        x2 = self.Bn_Cov2d_Relu1(x)

        y1 = torch.cat([x1, x2], dim=1)
        x3 = self.Bn_Cov2d_Relu2(y1)
        y2 = torch.cat([x1,x2,x3], dim=1)
        x4 = self.Bn_Cov2d_Relu3(y2)
        y3 = torch.cat([x1,x2,x3,x4], dim=1)
        x5 = self.Bn_Cov2d_Relu4(y3)
        y4 = torch.cat([x1,x2,x3,x4,x5], dim=1)
        out = self.Bn_Cov2d_Relu5(y4)
        # out = self.Conv1(out)
        # out = self.ReLU(out)
        # out = mean_normalization_tensor(out)
        return out

if __name__ == '__main__':
    input = torch.randn(1, 1, 128, 128)
    input2 = torch.randn(1, 1, 128, 128)
    input3 = torch.randn(1, 1, 128, 128)
    model = DenseBlock()
    output = model(input,input2)
    print(output.size())
