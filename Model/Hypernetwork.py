import torch
import cv2
import numpy as np
from Model.ImageProcessor import ImageProcessor
from Model.TemplateProcessor import TemplateProcessor


class HyperNetwork(torch.nn.Module):
    
    def __init__(self, image_processor : ImageProcessor, template_processor : TemplateProcessor, weight_path = None):
        super(HyperNetwork, self).__init__()
        if weight_path is not None:
            self.load_state_dict(torch.load(weight_path))
        self.image_processor = image_processor
        self.template_processor = template_processor
        self.upscaler = torch.nn.ConvTranspose2d(
            in_channels=template_processor.final_kernel_channels,
            out_channels=1,
            kernel_size=64,
            stride=32,
            padding=16,
            bias=False
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, image, template):
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if template.dim() == 3:
            template = template.unsqueeze(0)
        
        image_features = self.image_processor(image)
        template_kernel = self.template_processor(template)
        
        B,C,H,W = image_features.shape
        _,K,_,h,w = template_kernel.shape

        image_features = image_features.unsqueeze(1).repeat(1,K,1,1,1)
        
        reshaped_kernels = template_kernel.view(B*K,C,h,w)
        reshaped_images = image_features.view(1,B*K * C,H,W)

        heatmap = torch.nn.functional.conv2d(reshaped_images, reshaped_kernels, groups=B*K, padding=(h//2,w//2))
        heatmap = heatmap.view(B,K,heatmap.shape[2],heatmap.shape[3])
   
        heatmap = self.upscaler(heatmap)
        
        heatmap = torch.nn.functional.interpolate(heatmap, size=(image.shape[2],image.shape[3]), mode='bilinear', align_corners=False)
        
        heatmap = self.sigmoid(heatmap)
        
        heatmap = heatmap.squeeze(1)
        
        return heatmap
    
    def hash(self):
        return self.template_processor.hash
    
    def loss(self,heatmap,real_heatmap):
        eps = 1e-7
        heatmap = torch.clamp(heatmap, eps, 1.0 - eps)
        return torch.nn.functional.binary_cross_entropy(heatmap, real_heatmap)
        