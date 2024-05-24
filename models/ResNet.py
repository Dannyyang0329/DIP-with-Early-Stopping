import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=None)
        
        # Modify the first convolutional layer to accept the desired number of input channels
        self.resnet18.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Remove the fully connected layer and average pool layer
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])
        # Add a conv layer to match the output channel and use bilinear upsampling
        self.upsample = nn.ConvTranspose2d(512, output_channels, kernel_size=64, stride=32, padding=16)
        
    def forward(self, x):
        x = self.resnet18(x)
        x = self.upsample(x)
        return x
    