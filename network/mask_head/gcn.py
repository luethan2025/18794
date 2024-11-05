import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .conv import Conv1x1

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self._beta = Conv1x1(in_channels, out_channels)
        self._phi = Conv1x1(in_channels, out_channels)
        self._theta = Conv1x1(in_channels, out_channels)
  
    def forward(self, X):
        Z = F.softmax(np.dot(self._phi(X), self._theta(X)))
        Z = np.dot(self._beta, Z)
        return Z + X
