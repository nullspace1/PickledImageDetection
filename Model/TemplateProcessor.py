import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import time
import torchvision
from Model.Backbone import Backbone
class TemplateProcessor(nn.Module):
    
    def __init__(self, intermediate_size, final_kernel_channels=3):
        super().__init__()
        self.final_kernel_channels = final_kernel_channels
        self.backbone = Backbone()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(self.backbone.channels, intermediate_size)
        self.relu_1 = nn.ReLU()
        self.linear2 = nn.Linear(intermediate_size, final_kernel_channels * self.backbone.channels * 3 * 3)
        self.relu_2 = nn.ReLU()
        
        self.hash = hashlib.md5(str(intermediate_size + final_kernel_channels).encode()).hexdigest()[:5]
       

    def forward(self, x):
        features = self.backbone(x)           
        pooled = self.pool(features).view(features.size(0), -1) 
        x = self.relu_1(self.linear1(pooled))                        
        x = self.relu_2(self.linear2(x))                             
        x = x.view(x.size(0), self.final_kernel_channels, self.backbone.channels, 3, 3)
        return x
