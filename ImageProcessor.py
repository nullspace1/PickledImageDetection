import torch
import time
from Backbone import Backbone
import hashlib

class ImageProcessor(torch.nn.Module):
    
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.backbone = Backbone()
        self.hash = hashlib.md5(str(self.parameters()).encode()).hexdigest()[:5]
        
    def forward(self, x):
        start_time = time.time()
        result= self.backbone(x)
        print(f"Time taken in image processor: {time.time() - start_time:.3f} seconds")
        return result

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
