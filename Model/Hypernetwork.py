import os
import torch
import cv2
import numpy as np
from Model.ImageProcessor import ImageProcessor
from Model.TemplateProcessor import TemplateProcessor


class HyperNetwork(torch.nn.Module):
    
    def __init__(self, image_processor : ImageProcessor, template_processor : TemplateProcessor, weight_path = None):
        super(HyperNetwork, self).__init__()
        if weight_path is not None and os.path.exists(weight_path):
            self.load_state_dict(torch.load(weight_path))
        self.image_processor = image_processor
        self.template_processor = template_processor
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, image, template):
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if template.dim() == 3:
            template = template.unsqueeze(0)
        
        image_features = self.image_processor(image)
        template_kernel = self.template_processor(template)
        
        heatmap = torch.nn.functional.conv2d(image_features, template_kernel)
        heatmap = self.sigmoid(heatmap)

        return heatmap
    
    def hash(self):
        return self.template_processor.hash
    
    def loss(self,heatmap,real_heatmap):
        return torch.nn.functional.binary_cross_entropy(heatmap, real_heatmap)
    
        