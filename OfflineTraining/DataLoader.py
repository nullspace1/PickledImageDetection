import os
import numpy as np
import torch
import cv2
import random
from OfflineTraining.DataCreator import DataCreator
import time


class DataLoader:
    def __init__(self, data_save_path, data_creator):
        self.data_save_path = data_save_path
        self.data_creator = data_creator
        self.dataset = _Dataset(data_save_path, data_creator)
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=1,
            shuffle=True
        )
        self.iterator = iter(self.loader)

    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            raise StopIteration

class _Dataset(torch.utils.data.Dataset):
    def __init__(self, data_save_path, data_creator):
        self.data_save_path = data_save_path
        self.data_creator = data_creator
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.data_save_path):
            self.data_creator.create_data(self.data_save_path)
        self.data = np.load(self.data_save_path, allow_pickle=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        screenshot_file_link, heatmap_file_link, template_file_link = self.data[index]
        screenshot = np.load(screenshot_file_link)
        heatmap = np.load(heatmap_file_link)
        template = np.load(template_file_link)
        screenshot = torch.from_numpy(screenshot).float().permute(2, 0, 1) / 255
        template = torch.from_numpy(template).float().permute(2, 0, 1) / 255
        heatmap = torch.from_numpy(heatmap).float().permute(2, 0, 1) / 255
        heatmap = heatmap.mean(dim=0)
        return screenshot, template, heatmap
    

    
