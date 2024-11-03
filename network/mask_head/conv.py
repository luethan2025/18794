import torch.nn as nn

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        """1x1 convolution"""
        super(Conv1x1, self).__init__()
        self.kernel = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
    
    def forward(self, X):
        return self.kernel(X)

class Conv3x3(nn.Module):
    """3x3 convolution"""
    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()
        self.kernel = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False)
        )
  
    def forward(self, X):
        return self.kernel(X)
