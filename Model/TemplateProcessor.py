import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import time
import torchvision
from Model.Backbone import Backbone
class TemplateProcessor(nn.Module):

    def __init__(self, final_vector_size):
        super().__init__()
        self.final_vector_size = final_vector_size
        self.backbone = Backbone()
        self.kernel = nn.Conv2d(self.backbone.channels, self.final_vector_size, kernel_size=3, padding=1)
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.hash = hashlib.md5(str(self.final_vector_size).encode()).hexdigest()[:5]
       

    def forward(self, x):
        features = self.backbone(x)           
        vector = self.kernel(features)
        vector = self.average_pool(vector)
        return vector
    
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
