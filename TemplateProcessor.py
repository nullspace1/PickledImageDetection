import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import hashlib
class TemplateProcessor(nn.Module):
    
    def __init__(self, intermediate_size, final_kernel_channels=3):
        super().__init__()
        self.final_kernel_channels = final_kernel_channels
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(320, intermediate_size)
        self.relu_1 = nn.ReLU()
        self.linear2 = nn.Linear(intermediate_size, final_kernel_channels * 320 * 3 * 3)
        self.relu_2 = nn.ReLU()
       
        self.hash = hashlib.md5(str(self.parameters()).encode()).hexdigest()[:5]

    def forward(self, x):
        if x.shape[2] < 256 or x.shape[3] < 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        features = self.efficientnet(x)[-1]           
        pooled = self.pool(features).view(features.size(0), -1) 
        x = self.relu_1(self.linear1(pooled))                        
        x = self.relu_2(self.linear2(x))                             
        x = x.view(x.size(0), self.final_kernel_channels, 320, 3, 3)
        return x
