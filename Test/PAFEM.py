import torch
from torch import nn

class PAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(PAFEM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim),
                                       nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv2 = nnConv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
            # 如果batch=1 ，进行batchnorm会有问题
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        energy2 = torch.bmm(proj_query2, proj_key2)
        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2 * out2 + conv2
        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3
        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:],
                           mode='bilinear')  # 如果batch设为1，这里就会有问题。

        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))

