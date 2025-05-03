import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(224, 64, 3, stride=2, padding=1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dense_box = nn.Linear(64, 4)
        self.dense_found = nn.Linear(64, 1)
        
    def forward(self, correlation : Tensor) -> tuple[Tensor,Tensor]:
    
        correlation = self.conv2d(correlation)
        correlation = self.pooling(correlation)
        
        correlation = correlation.view(correlation.size(0), -1)

        box = F.relu(self.dense_box(correlation))
        found = F.sigmoid(self.dense_found(correlation))
        
        return box, found
    
    def test(self):
        print("Testing detection head...")
        test_input = torch.randn(1, 224, 128, 128)
        output = self.forward(test_input)
        print('output[0].shape: ', output[0].shape)
        print('expected: (1,4)')
        assert output[0].shape == (1,4)
        print('output[1].shape: ', output[1].shape)
        print('expected: (1,1)')
        assert output[1].shape == (1, 1)