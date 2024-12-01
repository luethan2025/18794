import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConvBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
 
class DownsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)

class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UpsampleConvBlock, self).__init__()
        if bilinear:
            self.upsample_block = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_block = DoubleConvBlock(in_channels, out_channels, mid_channels=in_channels//2)
        else:
            self.upsample_block = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2, padding=0)
            self.conv_block = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsample_block(x1)
        y = x2.shape[2] - x1.shape[2]
        x = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [x//2, x - x//2,
                        y//2, y - y//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, conv_dim=64, bilinear=False):
        super(UNet, self).__init__()
        self.double_conv = DoubleConvBlock(in_channels, conv_dim)

        self.d1 = DownsampleConvBlock(conv_dim, conv_dim*2)
        self.d2 = DownsampleConvBlock(conv_dim*2, conv_dim*4)
        self.d3 = DownsampleConvBlock(conv_dim*4, conv_dim*8)
        self.d4 = DownsampleConvBlock(conv_dim*8, conv_dim*16)

        scale_factor = 2 if bilinear else 1
        
        self.u1 = UpsampleConvBlock(conv_dim*16, (conv_dim*8)//scale_factor, bilinear=bilinear)
        self.u2 = UpsampleConvBlock(conv_dim*8, (conv_dim*4)//scale_factor, bilinear=bilinear)
        self.u3 = UpsampleConvBlock(conv_dim*4, (conv_dim*2)//scale_factor, bilinear=bilinear)
        self.u4 = UpsampleConvBlock(conv_dim*2, conv_dim, bilinear=bilinear)
    
        self.classifier = nn.Conv2d(conv_dim, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.double_conv(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)

        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)

        out = self.classifier(x)
        return out
