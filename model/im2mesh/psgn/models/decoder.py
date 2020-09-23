import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    r''' Simple decoder for the Point Set Generation Network.

    The simple decoder consists of 4 fully-connected layers, resulting in an
    output of 3D coordinates for a fixed number of points.

    Args:
        dim (int): The output dimension of the points (e.g. 3)
        bottleneck_size (int): dimension of the input vector
        n_points (int): number of output points
    '''
    def __init__(self, dim=3, bottleneck_size=128, n_points=1024):
        super().__init__()
        # Attributes
        self.dim = dim
        self.bottleneck_size = bottleneck_size
        self.n_points = n_points

        # Submodules
        self.actvn = F.relu
        self.fc_0 = nn.Linear(bottleneck_size, 512)
        self.fc_1 = nn.Linear(512, 512)
        self.fc_2 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, dim*n_points)

    def forward(self, c):
        batch_size = c.size(0)

        net = self.fc_0(c)
        net = self.fc_1(self.actvn(net))
        net = self.fc_2(self.actvn(net))
        points = self.fc_out(self.actvn(net))
        points = points.view(batch_size, self.n_points, self.dim)

        return points, c
