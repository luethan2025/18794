import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super(FCN, self).__init__()
        self._conv1 = nn.Conv2d(in_channels, intermediate_channels)
        self._conv2 = nn.Conv2d(intermediate_channels, out_channels)

    def forward(self, X):
        Z = F.relu(self._conv1(X))
        return self._conv2(Z)
