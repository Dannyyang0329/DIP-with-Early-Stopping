import torch
import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(
            self,
            num_of_layers: int=32,
            num_of_channels: int=256,
            kernel_size: int=3
        ):
        super(DeepCNN, self).__init__()
        assert num_of_layers >= 3, "Number of layers should be at least 3"

        padding = kernel_size // 2
        layers = [nn.Conv2d(1, num_of_channels, kernel_size, padding=padding), nn.PReLU()]
        for i in range(num_of_layers-2):
            layers.append(nn.Conv2d(num_of_channels, num_of_channels, kernel_size, padding=padding))
            layers.append(nn.PReLU())
        layers.append(nn.Conv2d(num_of_channels, 1, kernel_size, padding=padding))
        layers.append(nn.PReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return torch.squeeze(self.network(x.unsqueeze(0).unsqueeze(0)))
    