# import torch
# from torch import nn
#
# class Bn_Cov2d_Relu(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Bn_Cov2d_Relu, self).__init__()
#         self.seq = nn.Sequential(
#             nn.BatchNorm2d(in_channels),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         return self.seq(x)
#
# class spatial_attention(nn.Module):
#     def __init__(self):
#         super(spatial_attention, self).__init__()
#         kernel_size = 7
#         self.Conv1 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
#         self.Bn_Cov2d_Relu1 = Bn_Cov2d_Relu(64, 64)
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         x = self.Conv1(x)
#         x = self.Bn_Cov2d_Relu1(x)
#         max_pool_out,_ = torch.max(x,dim=1,keepdim=True)
#         mean_pool_out = torch.mean(x,dim=1,keepdim=True)
#         pol_out = torch.cat([max_pool_out,mean_pool_out],dim=1)
#         out = self.conv1(pol_out)
#         out = self.sigmoid(out)
#         return out * x
# if __name__ == '__main__':
#     x = torch.randn((1, 1, 640, 480))
#     model = spatial_attention()
#     out = model(x)
#     print(out.shape)