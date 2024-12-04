from torchvision import models

def import_backbone(name, pretrained=True):
    """Imports prebuild models from torchvision.
    Args:
        name (string): Name of backbone model (choices: ResNet18, ResNet50, VGG16, MobileNetV2).
    Returns:
        tuple (nn.Module, list, list): params that define the rest of the UNet model.
    """
    if name == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    elif name == 'vgg16':
        backbone = models.vgg16_bn(pretrained=pretrained).features
    elif name == 'mobilenetv2':
        backbone = models.mobilenet_v2(pretrained=pretrained).features
    else:
        raise ValueError(f'Unknown backbone: {name}')
    

    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    elif name == 'vgg16':
        feature_names = ['5', '12', '22', '32', '42']
        backbone_output = '43'
    elif name == 'mobilenetv2':
        feature_names = ['2', '4', '7', '14']
        backbone_output = '18'

    return backbone, feature_names, backbone_output
