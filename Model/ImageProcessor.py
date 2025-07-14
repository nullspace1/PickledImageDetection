import torch
import time
from Model.Backbone import Backbone
import hashlib
import torch.nn as nn
import torch.nn.functional as F

class ImageProcessor(torch.nn.Module):
    
    def __init__(self, final_vector_size, grid_size, shrink_factor : int = 16):
        super(ImageProcessor, self).__init__()
        self.backbone = Backbone()
        self.kernel = nn.Conv2d(self.backbone.channels, final_vector_size, kernel_size=3, padding=1)
        self.average_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.final_vector_size = final_vector_size
        self.shrink_factor = shrink_factor
        self.grid_size = grid_size
        self.hash = hashlib.md5(str(final_vector_size + grid_size + shrink_factor).encode()).hexdigest()[:5]
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=1/self.shrink_factor, mode='bilinear', align_corners=False)
        features = self.backbone(x)           
        vector = self.kernel(features)
        vector = self.average_pool(vector).view(vector.size(0), -1) 
        vector = vector.view(vector.size(0), self.final_vector_size, self.grid_size, self.grid_size)
        return vector

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
