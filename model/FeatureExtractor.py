import torch
import torch.nn as nn
import torchvision
from torch import Tensor


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.normalize_input = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
    def forward(self, screenshot : Tensor, crop : Tensor) -> tuple[tuple[Tensor,Tensor,Tensor], tuple[Tensor,Tensor,Tensor]]:
        
        print("Extracting features...")
        
        screenshot = screenshot.permute(0, 3, 1, 2)
        crop = crop.permute(0, 3, 1, 2)
        
        screenshot = self.normalize_input(screenshot/255)
        crop = self.normalize_input(crop/255)
        
        sfeat1 : torch.Tensor = self.conv1(screenshot)
        sfeat2 : torch.Tensor = self.conv2(sfeat1)
        sfeat3 : torch.Tensor  = self.conv3(sfeat2)
        
        cfeat1 : torch.Tensor  = self.conv1(crop)
        cfeat2 : torch.Tensor  = self.conv2(cfeat1)
        cfeat3 : torch.Tensor  = self.conv3(cfeat2)
        
        return (sfeat3, sfeat2, sfeat1), (cfeat3, cfeat2, cfeat1)
    
