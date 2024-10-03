import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
'''
    This is SENet test.
'''
class extension_channels(nn.Module):
    def __init__(self):
        super(extension_channels, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        h,w,c = x.shape
        b = 1
        x = x.reshape(b,c,h,w)
        x = self.conv1(x)
        return x
class SEnet(nn.Module):
    def __init__(self, input_channel, ratio = 16):
        super(SEnet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_channel, int(ratio*input_channel)),
            nn.ReLU(),
            nn.Linear(int(ratio*input_channel), input_channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,h,w = x.size()
        avg = self.avg_pool(x).view(b,c)
        fc = self.fc(avg).view(b,c,1,1)
        return x*fc
class shrinkage_channels(nn.Module):
    def __init__(self):
        super(shrinkage_channels, self).__init__()
        self.conv1 = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        return x
if __name__ == '__main__':
    image_path = "../Datasets/train/vi/00001D.png"
    image = torch.tensor(mpimg.imread(image_path))

    extension_channels = extension_channels()
    image = extension_channels(image)

    shrinkage_channels = shrinkage_channels()
    image = shrinkage_channels(image)

    model = SEnet(input_channel=3)
    image = model(image)*255
    image = image.squeeze(0).permute(1,2,0).detach().numpy()
    plt.imshow(image)
    plt.show()
    print(image.shape)
