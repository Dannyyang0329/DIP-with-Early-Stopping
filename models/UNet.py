import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # input example: (a x a x in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)     # (a-2 x a-2 x out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)    # (a-4 x a-4 x out_channels)
        self.bn = nn.BatchNorm2d(out_channels)                              # (a-4 x a-4 x out_channels)
        self.relu = nn.ReLU(inplace=True)                                   # (a-4 x a-4 x out_channels)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn(self.conv2(x))
        x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Down-sampling layers
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)            
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Up-sampling layers
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv9 = ConvBlock(512+256, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = ConvBlock(256+128, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7 = ConvBlock(128+64, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv6 = ConvBlock(64+64, 64)
        self.conv5 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Down-sampling path
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.pool4(x4)

        # Up-sampling path
        x = self.upconv4(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv9(x)
        x = self.upconv3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv8(x)
        x = self.upconv2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv7(x)
        x = self.upconv1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv6(x)
        x = self.conv5(x)
        # x = nn.ReLU(inplace=True)(x)

        return x