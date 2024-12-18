import torch
import torch.nn as nn
from torchvision.models import resnet18

#Create Model class:
class ResnetsegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(ResnetsegmentationModel, self).__init__()
        base_model = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        
    def forward(self,x):
        features = self.backbone(x)
        out = self.classifier(features)
        out = nn.functional.interpolate(out, scale_factor=32, mode='bilinear', align_corners=False)
        return out
    
    