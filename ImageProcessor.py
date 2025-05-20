import torch
import timm

class ImageProcessor(torch.nn.Module):
    
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
    def forward(self, x):
        return self.efficientnet(x)[-1]

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
