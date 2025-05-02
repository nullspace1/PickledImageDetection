import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F


class CrossCorrelation(nn.Module):
    def __init__(self):
        super().__init__()
        
    def cross_correlate(self, screen_feat: Tensor, template_feat: Tensor) -> Tensor:
        template_feat = torch.flip(template_feat, [2,3])
        
        B,C,H,W = screen_feat.shape
        _,_,h,w = template_feat.shape
        
        screen_feat = screen_feat.reshape(1,B*C,H,W)
        template_feat = template_feat.reshape(B*C,1,h,w)
        
        print("Running cross correlation...")

        out = F.conv2d(screen_feat, template_feat, bias=None, stride=1, padding=0)
        
        out = out.reshape(B, C, out.shape[-2], out.shape[-1])

        return out
               
    def forward(self, screenshot_feats : tuple[Tensor,Tensor,Tensor], template_feats : tuple[Tensor,Tensor,Tensor]) -> Tensor:
        
        print("Computing cross correlation...")
        
        correlation_1 : Tensor = self.cross_correlate(screenshot_feats[0], template_feats[0])
        correlation_2 : Tensor = self.cross_correlate(screenshot_feats[1], template_feats[1])
        correlation_3 : Tensor = self.cross_correlate(screenshot_feats[2], template_feats[2])
        
        correlation_1 : Tensor = F.interpolate(correlation_1, (128, 128))
        correlation_2 : Tensor = F.interpolate(correlation_2, (128, 128))
        correlation_3 : Tensor = F.interpolate(correlation_3, (128, 128))
        
        correlation : Tensor = torch.cat([correlation_1, correlation_2, correlation_3], dim=1)
        
        return correlation