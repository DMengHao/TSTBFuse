# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class DenseBlock(nn.Module):
#     def __init__(self, in_channels, growth_rate):
#         super(DenseBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(4 * growth_rate)
#         self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate-3, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(growth_rate-3)
#         self.conv3 = nn.Conv2d(1, 3, kernel_size=1, padding=0, bias=False)
#     def forward(self, x):
#         if x.size(1)==1:
#             z = self.conv3(x)
#             out = self.conv1(x)
#             out = self.bn1(out)
#             out = F.relu(out)
#             out = self.conv2(out)
#             out = self.bn2(out)
#             out = F.relu(out)
#             out = torch.cat([z, out], 1)
#         elif x.size(1)==3:
#             z = x
#             out = self.conv1(x)
#             out = self.bn1(out)
#             out = F.relu(out)
#             out = self.conv2(out)
#             out = self.bn2(out)
#             out = F.relu(out)
#             out = torch.cat([z, out], 1)
#         return out
#
#
# class Decoder(nn.Module):
#     def __init__(self, in_channels=64, out_channels=1):
#         super(Decoder, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(16)
#         self.conv_restore = nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0)
#     def forward(self, x):
#         x = self.bn1(self.relu(self.conv1(x)))
#         x = self.bn2(self.relu(self.conv2(x)))
#         x = self.conv_restore(x)
#         return x
#
#
# # 定义模型结构
# class ThreeInputsNet(nn.Module):
#     def __init__(self):
#         super(ThreeInputsNet, self).__init__()
#         # 3, 64, 64
#         self.conv1_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.pooling1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv1_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv1_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv1_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         # 3, 64, 64
#         self.conv2_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.pooling2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         # 3, 64, 64
#         self.conv3_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.pooling3_1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv3_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv3_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         # 128, 4, 4
#         # 三个通道的channel合并
#         # 128*5, 4, 4
#         self.outlayer1 = nn.Linear(3 * 128 * 4 * 4, 128 * 5)
#         self.outlayer2 = nn.Linear(128 * 5, 256)
#         self.outlayer3 = nn.Linear(256, 5)  # 是几分类第二个数就改成几，比如我们做5分类的任务，这里就是5
#
#     # 此处的输入为三个，对应三个分支
#     def forward(self, input1, input2, input3):
#         out1 = self.pooling1_1(self.conv1_1(input1))
#         out1 = self.pooling1_1(self.conv1_2(out1))
#         out1 = self.pooling1_1(self.conv1_3(out1))
#         out1 = self.pooling1_1(self.conv1_4(out1))
#
#         out2 = self.pooling2_1(self.conv2_1(input2))
#         out2 = self.pooling2_1(self.conv2_2(out2))
#         out2 = self.pooling2_1(self.conv2_3(out2))
#         out2 = self.pooling2_1(self.conv2_4(out2))
#
#         out3 = self.pooling3_1(self.conv3_1(input3))
#         out3 = self.pooling3_1(self.conv3_2(out3))
#         out3 = self.pooling3_1(self.conv3_3(out3))
#         out3 = self.pooling3_1(self.conv3_4(out3))
#         # 将三个分支的结果在channel维度上合并
#         out = torch.cat((out1, out2, out3), dim=1)
#         out = out.view(out.size(0), -1)  # [B, C, H, W] --> [B, C*H*W]
#         out = self.outlayer1(out)
#         out = self.outlayer2(out)
#         out = self.outlayer3(out)
#         return out
#
