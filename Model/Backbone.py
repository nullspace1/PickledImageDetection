import torch.nn as nn
import timm
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True, features_only=True)
        self.layer_index = 3
        self.channels = 256

    def forward(self, x):
        if x.shape[2] < 256 or x.shape[3] < 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return self.backbone(x)[self.layer_index]
