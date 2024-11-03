import torch.nn as nn

from .conv import Conv1x1, Conv3x3
from .gcn import GCN
from .fcn import FCN
from .upsampling import UpsampleConv

class MaskHead(nn.Module):
    def __init__(
            self, conv3x3_in_channels, conv3x3_out_channels, gcn_in_channels, 
            gcn_out_channels, fcn_in_channels, fcn_intermediate_channels,
            fcn_out_channels, upsample_in_channels, upsample_out_channels,
            conv1x1_in_channels, conv1x1_out_channels):
        self._occluder_perception_branch = nn.Sequential(
            Conv3x3(conv3x3_in_channels, conv3x3_out_channels),
            GCN(gcn_in_channels, gcn_out_channels),
            FCN(fcn_in_channels, fcn_intermediate_channels, fcn_out_channels)
        )

        self._occludee_perception_branch = nn.Sequential(
            Conv3x3(conv3x3_in_channels, conv3x3_out_channels),
            GCN(gcn_in_channels, gcn_out_channels),
            FCN(fcn_in_channels, fcn_intermediate_channels, fcn_out_channels)
        )

        self._upsample = UpsampleConv(upsample_in_channels, upsample_out_channels)
        self._pointwise_conv = Conv1x1(conv1x1_in_channels, conv1x1_out_channels)
  
    def forward(self, roi_feature):
        Z0 = self._occluder_perception_branch(roi_feature)
        Z1 = self._occludee_perception_branch(roi_feature + Z0)
        Z0_occ_B = self._upsample(Z0)
        Z0_occ_S = self._pointwise_conv(Z0)
        Z1_occ_B = self._upsample(Z1)
        Z1_occ_S = self._pointwise_conv(Z1)
        return Z0_occ_B, Z0_occ_S, Z1_occ_B, Z1_occ_S
        