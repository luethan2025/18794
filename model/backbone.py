from torch import nn
from torchvision import models

class FCOSHead(nn.Module):
    def __init__(self):
        self.backbone = models.detection.fcos_resnet50_fpn(weights= models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT)
        
    def forward(self,images,annotations):
        return self.backbone.forward(images,annotations)
