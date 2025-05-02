import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.pooling = nn.AvgPool2d(3, stride=2, padding=1)
        self.dense_box = nn.Linear(64, 4)
        self.dense_found = nn.Linear(64, 1)
        
    def forward(self, correlation : Tensor) -> tuple[Tensor,Tensor]:
        
        print("Running detection head...")
        
        correlation = self.conv2d(correlation)
        correlation = self.pooling(correlation)
        
        box = F.relu(self.dense_box(correlation))
        found = F.sigmoid(self.dense_found(correlation))
        
        return box, found