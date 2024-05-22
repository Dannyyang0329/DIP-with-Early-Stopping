import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(
            self,
            num_of_channel: int=1,
            features: int=64,
            num_of_layers: int=32,
            kernel_size: int=3,
            padding: int=1
        ):
        super(DnCNN, self).__init__()

        layers = [
            nn.Conv2d(num_of_channel, features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        ]
        for i in range(num_of_layers-2):
            layers.append(nn.Conv2d(features, features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, num_of_channel, kernel_size, padding=padding, bias=False))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out
    