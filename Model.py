import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convolutional_1 = torch.nn.Conv2d(1, 32, 3, 1, 1)
        self.relu_1 = torch.nn.ReLU()
        self.convolutional_2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.relu_2 = torch.nn.ReLU()
        self.max_pooling_1 = torch.nn.MaxPool2d(2, 2)
        self.flatten = torch.nn.Flatten()
        self.linear_1 = torch.nn.Linear(64 * 32 * 32, 30)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
            x = x.permute(0, 3, 1, 2)

        x = self.convolutional_1(x)
        x = self.relu_1(x)
        x = self.convolutional_2(x)
        x = self.relu_2(x)
        x = self.max_pooling_1(x)
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def loss(self, x, y):
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        similarity = torch.nn.functional.cosine_similarity(x_norm, y_norm)
        return 1 - similarity

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_state_dict'] is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

