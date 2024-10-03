import torch
from torch import nn
from einops import rearrange
import numbers
import torch.nn.functional as F


def mean_normalization_tensor(data):
    mean = torch.mean(data, dim=(1, 2, 3), keepdim=True)  # 计算均值
    std = torch.std(data, dim=(1, 2, 3), keepdim=True)    # 计算标准差
    normalized_data = (data - mean) / std  # 均值归一化
    return normalized_data
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


# 归一化处理
# 每个样本的特征进行均值为 0、方差为 1 的标准化处理
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    # 64 2 False
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        #  64 256 1 False
        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 256 256 3 1 1 256 False
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        # 128 64 1 False
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# Multi_Attention(Q,K,V) 输出是经过自注意力加权后的特征张量，具有与输入相同的维度和形状。
class Attention(nn.Module):
    # 自注意力模块的作用是对输入特征图进行自注意力计算，从而获取每个位置的重要程度，并生成对应的输出特征。
    # 64 8 Flase
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # 8行一列 全1 温度参数来调节注意力的尖峰度
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 64 64*3 1 Flase
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 64*3 64*3 3 1 1 64*3 False
        # groups=64*3 每个输入通道连接对应的输出通道不会连接到其他的输出通道
        # 用来捕获特征的局部相关性
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 64 64 1 False
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # (8 64 128 128)
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        # 64 64 64
        q, k, v = qkv.chunk(3, dim=1)
        # (8 (8 192) 128 128)->(8 8 192 (128 128))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
class TransformerBlock(nn.Module):
    # 64 8 2 False WithBias
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        # 每个样本的特征进行均值为 0、方差为 1 的标准化处理
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # 64 8 False 输出是经过自注意力加权后的特征张量，具有与输入相同的维度和形状。64 8 False
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # 64 2 False 前馈神经网络其作用是对输入特征进行非线性变换
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # Attention(Q,K,V) 输出是经过自注意力加权后的特征张量，具有与输入相同的维度和形状。
        x = x + self.attn(self.norm1(x))
        #  前馈神经网络 其作用是对输入特征进行非线性变换
        x = x + self.ffn(self.norm2(x))
        return x

class Restormer_Decoder(nn.Module):
    def __init__(self,
                 # 输入通道数
                 inp_channels=1,
                 # 输出通道数
                 out_channels=1,
                 dim=128,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid = nn.Sigmoid()
        self.reduce = nn.Conv2d(int(dim*1.5), int(dim), kernel_size=1, bias=bias)

    def forward(self, inp_img,ir_vi_feature, ir_feature, vi_feature):
        if vi_feature==None:
            ir_features = torch.cat((ir_vi_feature, ir_feature), dim=1)
            # ir_features = self.vi_Conv(ir_features)
            ir_features = self.encoder_level2(ir_features)
            if inp_img is not None:
                inp_img = inp_img-0.5
                ir = self.output(ir_features) + inp_img
            else:
                ir = self.output(ir_features)
            return self.sigmoid(ir+0.5)
        elif ir_feature==None:
            vi_features = torch.cat(( ir_vi_feature, vi_feature), dim=1)
            # vi_features = self.vi_Conv(vi_features)
            vi_features = self.encoder_level2(vi_features)
            if inp_img is not None:
                inp_img = inp_img - 0.5
                vi = self.output(vi_features) + inp_img
            else:
                vi = self.output(vi_features)
            return self.sigmoid(vi+0.5)
        else:
            out_enc_level0 = torch.cat((ir_vi_feature, ir_feature, vi_feature), dim=1)
            out_enc_level0 = self.reduce(out_enc_level0)
            out_enc_level1 = self.encoder_level2(out_enc_level0)
            # 使用残差结构
            if inp_img is not None:
                inp_img = inp_img - 0.5
                out_enc_level1 = self.output(out_enc_level1) + inp_img
            else:
                out_enc_level1 = self.output(out_enc_level1)
            return self.sigmoid(out_enc_level1+0.5), out_enc_level0+0.5
if __name__ == '__main__':
    x = torch.rand(1, 1, 640, 480)
    x1 = torch.rand(1, 1, 640, 480)
    x2 = torch.rand(1, 63, 640, 480)
    x3 = torch.rand(1, 192, 640, 480)
    Restormer_Decoder = Restormer_Decoder()
    out, _ = Restormer_Decoder(x, x1, x2, x3)
    print(out.shape)
    print(out.size())
    print(out.max().item())
