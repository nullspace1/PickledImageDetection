import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class DetectionHead(nn.Module):
    def __init__(self, feature_extractor_out_channels : tuple[int,int,int], conv_out_channels : int, pool_size : tuple[int,int]):
        super().__init__()
        self.out_channels = feature_extractor_out_channels
        self.sum_channels = feature_extractor_out_channels[0] + feature_extractor_out_channels[1] + feature_extractor_out_channels[2]
        self.conv2d = nn.Conv2d(self.sum_channels, conv_out_channels, 3, stride=2, padding=1)
        self.pooling = nn.AdaptiveAvgPool2d(pool_size)
        self.dense_box = nn.Linear(pool_size[0] * pool_size[1] * conv_out_channels, 4)
        self.dense_found = nn.Linear(pool_size[0] * pool_size[1] * conv_out_channels, 1)
        
    def forward(self, correlation : Tensor) -> tuple[Tensor,Tensor]:
    
        correlation = self.conv2d(correlation)
        correlation = self.pooling(correlation)
        
        # turn [B,C,H,W] to [B,C*H*W]
        correlation = correlation.view(correlation.size(0), -1)

        box = F.relu(self.dense_box(correlation))
        found = F.sigmoid(self.dense_found(correlation))
        
        return box, found
    
    def test(self):
        print("Testing detection head...")
        test_input = torch.randn(1, self.sum_channels, 128, 128)
        output = self.forward(test_input)
        print('output[0].shape: ', output[0].shape)
        print('expected: (1,4)')
        assert output[0].shape == (1,4)
        print('output[1].shape: ', output[1].shape)
        print('expected: (1,1)')
        assert output[1].shape == (1, 1)