import cProfile
import torch
import torch.ao.quantization
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.nn import functional as F
from torchao.quantization import (  
    quantize_,  
    int4_weight_only,  
)
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
from torch.ao.quantization import quantize_dynamic
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
import torchao

def get_backbone(name, pretrained=True):

    """ Loading backbone, defining names for skip-connections and encoder output. """

    # TODO: More backbones

    # loading backbone model
    #torch.backends.quantized.engine = 'qnnpack'
    if(name == 'mobilenet'):
        print("hi")
        backbonetmo =  models.quantization.mobilenet_v2(weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1, quantize=True,progress=True)
        backbone = backbonetmo.features
        quant = backbonetmo.quant
        dequant = backbonetmo.dequant
        jit_list = []
        for name3, child in backbone.named_children():
            jit_list.append((name3, torch.jit.script(child)))
    elif name == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained =pretrained)
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        backbone = models.resnet152(pretrained=pretrained)
    elif name == 'vgg16':
        backbone = models.vgg16_bn(pretrained=pretrained).features
    elif name == 'vgg19':
        backbone = models.vgg19_bn(pretrained=pretrained).features
    # elif name == 'inception_v3':
    #     backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
    elif name == 'densenet121':
        backbone = models.densenet121(pretrained=True).features
    elif name == 'densenet161':
        backbone = models.densenet161(pretrained=True).features
    elif name == 'densenet169':
        backbone = models.densenet169(pretrained=True).features
    elif name == 'densenet201':
        backbone = models.densenet201(pretrained=True).features
    elif name == 'unet_encoder':
        from unet_backbone import UnetEncoder
        backbone = UnetEncoder(3)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names

    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    elif name == 'mobilenet':
        print("hi2")
        feature_names = [None, '2', '6', '11']
        backbone_output = '18'
    elif name == 'vgg16':
        # TODO: consider using a 'bridge' for VGG models, there is just a MaxPool between last skip and backbone output
        feature_names = ['5', '12', '22', '32', '42']
        backbone_output = '43'
    elif name == 'vgg19':
        feature_names = ['5', '12', '25', '38', '51']
        backbone_output = '52'
    # elif name == 'inception_v3':
    #     feature_names = [None, 'Mixed_5d', 'Mixed_6e']
    #     backbone_output = 'Mixed_7c'
    elif name.startswith('densenet'):
        feature_names = [None, 'relu0', 'denseblock1', 'denseblock2', 'denseblock3']
        backbone_output = 'denseblock4'
    elif name == 'unet_encoder':
        feature_names = ['module1', 'module2', 'module3', 'module4']
        backbone_output = 'module5'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone, feature_names, backbone_output, quant, dequant, jit_list


class Conv2dSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(Conv2dSeparable, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                      stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.conv(x)

class Conv2dSeparableTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=False):
        super(Conv2dSeparableTranspose, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, output_padding=output_padding, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                      stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.conv(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, skip_in=0, use_bn=True, parametric=False, use_sc=False):
        super(UpsampleBlock, self).__init__()
        #print(in_channels, out_channels)

        self.parametric = parametric
        out_channels = (in_channels / 2) if out_channels is None else out_channels

        if parametric:
            if not use_sc:
                self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(4, 4),
                                            stride=2, padding=1, output_padding=0, bias=(not use_bn))
            else:
                self.up = Conv2dSeparableTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=(4, 4),
                                                   stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else None
        else:
            self.up = None
            in_channels = in_channels + skip_in # concatenate skip connection channels
            if not use_sc:
                self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                       stride=1, padding=1, bias=(not use_bn))
            else:
                self.conv1 = Conv2dSeparable(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                             stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        conv2_in = out_channels if not parametric else (out_channels + skip_in)
        if not use_sc:
            self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=out_channels, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
        else:
            self.conv2 = Conv2dSeparable(in_channels=conv2_in, out_channels=out_channels, kernel_size=(3, 3),
                                         stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else None

    def forward(self, x, skip_connection=None):
        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            
            x = torch.cat([x, skip_connection.cuda()], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x

class Goon(nn.Module):
    """U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones.
    """
    def __init__(self,
                 backbone_name='mobilenet',
                 pretrained=True,
                 encoder_freeze=False,
                 classes=21,
                 decoder_filters=(96, 32, 24, 16, 18),
                 parametric_upsampling=True,
                 shortcut_features='default',
                 decoder_use_batchnorm=True,
                 use_separable_conv=False,
                 quantize = False):
        super(Goon, self).__init__()
        self.backbone_name = backbone_name
        self.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
        self.quant2 = torch.ao.quantization.QuantStub()
        self.dequant2 = torch.ao.quantization.DeQuantStub()
        self.quantize = quantize
        self.backbone, self.shortcut_features, self.bb_out_name,self.quant, self.dequant, self.jit_list = get_backbone(backbone_name, pretrained=pretrained)
        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        #print("mon", shortcut_chs, bb_out_chs, "key")
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[:len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_batchnorm,
                                                      use_sc=use_separable_conv))
        # self.model_conv = nn.Conv2d(1280, 512, kernel_size=1)
        if not use_separable_conv:
            self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))
        else:
            self.final_conv = Conv2dSeparable(decoder_filters[-1], classes, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()

        self.replaced_conv1 = False  # for accommodating inputs with different number of channels later

    def freeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input):
        x, features = self.forward_backbone(*input)
        x = self.dequant(x)
        if(self.quantize):
            x = self.quant2(x)
        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = x.cuda()
            x = upsample_block(x, skip_features)
        if self.backbone_name == 'mobilenet':
            x = F.interpolate(x, size=None, scale_factor=2.0, mode='nearest', align_corners=None)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        features = {None: None} if None in self.shortcut_features else dict()
        x = self.quant(x.to('cpu'))
        for name, child in self.jit_list:
            #print(name,child)
            #print("AHHHH", name, type(x), type(x[0, 0, 0, 0]), x[0, 0, 0, 0], self.bb_out_name)
            x = child(x)
            if name in self.shortcut_features:
                tmp = self.dequant(x)
                if(self.quantize):
                    tmp = self.quant2(tmp)
                features[name] = tmp
            if name == self.bb_out_name:
                break
        #x = self.model_conv(x)
        #print(x.shape)
        
        return x, features

    def infer_skip_channels(self):
        tmp = self.quant
        x = torch.empty(1,3,224,224).normal_()
        x = tmp(x)
        print(x.is_quantized)
        count = 0
        has_fullres_features = self.backbone_name.startswith('vgg')
        channels = [] if has_fullres_features else [0]  # only VGG has features at full resolution
        # print(list(self.backbone.named_children()))

        for name, child in self.jit_list:
            #print(name, child)
            
            #print("AHHHH", name, type(x), type(x[0, 0, 0, 0]), x[0, 0, 0, 0], self.bb_out_name)
            if(name == 'quant'):
                continue
            x = child(x)
            count += 1
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



if __name__ == "__main__":

    # simple test run
    #torch.jit.enable_onednn_fusion(True)
    #print(torch.backends.quantized.engine)
    batch_tmp = torch.empty(1,3,224,224).normal_()
    torch.backends.quantized.engine = 'qnnpack'
    tmp = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
    res = tmp(batch_tmp)
    net = UNet(backbone_name='mobilenet', quantize=False)
    #quantize_(mode, int4_weight_only())
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)
    # prepare and convert model
    # Set the backend on which the quantized kernels need to be run
  
 
    #net(batch_tmp)
    net.eval()
    #net2.eval()
    net2= net
    #net2=  torch.compile(net, mode= "max-autotune")
    #net2.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    #net2.eval()
    
    #print(net2)
    #net2.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')

    #new_upsmaple_blocks =nn.ModuleList()
    #for layer in net.upsample_blocks:
    #    new_upsmaple_blocks.append(torch.ao.quantization.fuse_modules(layer, [['conv2', 'bn2','relu']]))
    #net2.upsample_blocks = new_upsmaple_blocks
    #pytorch_total_params = sum(p.numel() for p in net2.parameters())
    #print(pytorch_total_params)
    #model_fp32_prepared = torch.ao.quantization.prepare(net2)
    #model_fp32_prepared(batch_tmp)
    #model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
    #pytorch_total_params = sum(p.numel() for p in model_int8.parameters())
    print(pytorch_total_params)
    #net2 = torch.jit.script(net)
    #net2.eval()
    net2.zero_grad()
    import time
    criterion = nn.MSELoss()
    print("goon")
    optimizer = torch.optim.Adam(net2.parameters())
    print('Network initialized. Running a test batch.')
    print(f"thread count = {torch.get_num_threads()}")
    start = time.time()

    for _ in range(20):   
        with torch.set_grad_enabled(True):
            batch = torch.ones(1, 3, 224, 224).normal_()
            targets = torch.ones(1, 21, 224, 224).normal_()
            start = time.time()
            out = net2(batch)
            #print(out)
            end = time.time()
            print(out.shape)
            #loss = criterion(out, targets)
            #loss.backward()
            #optimizer.step()
         #print(out.shape)
    print(f"elapsed time = {end - start}")
    print('fasza.')

