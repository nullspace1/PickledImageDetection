import torch.nn as nn
from torch import Tensor
from model.FeatureExtractor import FeatureExtractor
from model.CrossCorrelation import CrossCorrelation
from model.DetectionHead import DetectionHead

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.cross_correlation = CrossCorrelation()
        self.detection_head = DetectionHead()
        
    def forward(self, screenshot : Tensor, crop : Tensor) -> tuple[Tensor,Tensor]:
        screenshot_features, crop_features = self.feature_extractor.forward(screenshot, crop)
        correlation = self.cross_correlation.forward(screenshot_features, crop_features)
        box, found = self.detection_head.forward(correlation)
        return box, found