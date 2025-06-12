import torch
import time
from Model.Backbone import Backbone
import hashlib

class ImageProcessor(torch.nn.Module):
    
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.backbone = Backbone()
        
    def forward(self, x):
        result= self.backbone(x)
        return result

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
