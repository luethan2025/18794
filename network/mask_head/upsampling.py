import torch.nn as nn

class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        self._upsample = nn.ConvTranspose2d(in_channels, in_channels)
        self._conv = nn.Conv2d(in_channels, out_channels)

    def forward(self, X):
        return self._conv(self._upsample(X))
