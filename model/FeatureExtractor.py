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
        
        screenshot = screenshot.permute(0, 3, 1, 2)
        crop = crop.permute(0, 3, 1, 2)
        
        print(screenshot.shape)
        
        screenshot = self.normalize_input(screenshot/255)
        crop = self.normalize_input(crop/255)
        
        sfeat1 : torch.Tensor = self.conv1(screenshot)
        sfeat2 : torch.Tensor = self.conv2(sfeat1)
        sfeat3 : torch.Tensor  = self.conv3(sfeat2)
        
        cfeat1 : torch.Tensor  = self.conv1(crop)
        cfeat2 : torch.Tensor  = self.conv2(cfeat1)
        cfeat3 : torch.Tensor  = self.conv3(cfeat2)
        
        return (sfeat3, sfeat2, sfeat1), (cfeat3, cfeat2, cfeat1)
    
    def test(self) -> None:
        input_screen = torch.randn(1, 1024, 512, 3)
        input_crop = torch.randn(1, 1024, 512,3)
        out = self.forward(input_screen, input_crop)
        (sfeat3, sfeat2, sfeat1), (cfeat3, cfeat2, cfeat1) = out
        print('sfeat3.shape: ', sfeat3.shape)
        print('expected: (1, 128, 256, 128)')
        assert sfeat3.shape == (1, 128, 1024/8, 512/8)
        
        print('sfeat2.shape: ', sfeat2.shape)
        print('expected: (1, 64, 512, 256)')
        assert sfeat2.shape == (1, 64, 1024/4, 512/4)

        print('sfeat1.shape: ', sfeat1.shape)
        print('expected: (1, 32, 1024, 512)')
        assert sfeat1.shape == (1, 32, 1024/2, 512/2)

        print('cfeat3.shape: ', cfeat3.shape)
        print('expected: (1, 128, 128, 64)')
        assert cfeat3.shape == (1, 128, 1024/8, 512/8)

        print('cfeat2.shape: ', cfeat2.shape)
        print('expected: (1, 64, 256, 128)')
        assert cfeat2.shape == (1, 64, 1024/4, 512/4)

        print('cfeat1.shape: ', cfeat1.shape)
        print('expected: (1, 32, 512, 256)')
        assert cfeat1.shape == (1, 32, 1024/2, 512/2)
        