import torch
import torch.nn as nn
from torch.nn import functional as F
from .backbones import import_backbone

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        out_channels = (in_channels / 2) if out_channels is None else out_channels

        if parametric:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else None
        else:
            self.up = None
            in_channels = in_channels + skip_in # concatenate skip connection channels
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                    stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else None

        conv2_in = out_channels if not parametric else (out_channels + skip_in)
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=out_channels, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else None

    def forward(self, x, skip_connection=None):
        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2.0, mode='nearest', align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = nn.functional.relu(x, inplace=True)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = nn.functional.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = nn.functional.relu(x, inplace=True)

        return x

class UNet(nn.Module):
    """U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones.
    """
    def __init__(self,
                 backbone_name='resnet18',
                 pretrained=True,
                 encoder_freeze=False,
                 classes=21,
                 decoder_filters=(256, 128, 64, 32, 16),
                 parametric_upsampling=True,
                 shortcut_features='default',
                 decoder_use_batchnorm=True):
        super(UNet, self).__init__()
        self.backbone_name = backbone_name

        self.backbone, self.shortcut_features, self.bb_out_name = import_backbone(backbone_name, pretrained=pretrained)
        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        if self.backbone_name == 'mobilenetv2':
            decoder_filters = [96, 32, 24, 16, 18]
        decoder_filters = decoder_filters[:len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_batchnorm))

        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()

        self.replaced_conv1 = False  # for accommodating inputs with different number of channels later

    def freeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input):
        x, features = self.forward_backbone(*input)
        for (skip_name, upsample_block) in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)
        if self.backbone_name == 'mobilenetv2':
            x = F.interpolate(x, size=None, scale_factor=2.0, mode='nearest', align_corners=None)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break
        return x, features

    def infer_skip_channels(self):
        x = torch.zeros(1, 3, 224, 224)
        has_fullres_features = self.backbone_name.startswith('vgg')
        channels = [] if has_fullres_features else [0]  # only VGG has features at full resolution
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels

    def get_pretrained_parameters(self):
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                yield param

    def get_random_initialized_parameters(self):
        pretrained_param_names = set()
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                pretrained_param_names.add('backbone.{}'.format(name))

        for name, param in self.named_parameters():
            if name not in pretrained_param_names:
                yield param
