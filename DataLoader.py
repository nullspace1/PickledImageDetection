import os
import numpy as np
import torch
import cv2
import random
from DataCreator import DataCreator
import time


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data_save_path, data_creator, batch_size=1):
        self.data_save_path = data_save_path
        self.data_creator = data_creator
        self.batch_size = batch_size
        self.load_data()


    def load_data(self):
        if not os.path.exists(self.data_save_path):
            self.data_creator.create_data(self.data_save_path)
        self.data = np.load(self.data_save_path, allow_pickle=True)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        screenshot_file_link, heatmap_file_link, template_file_link  = self.data[index]
        screenshot = torch.load(screenshot_file_link)
        heatmap = torch.load(heatmap_file_link)
        template = torch.load(template_file_link)
        return screenshot,  template, heatmap

    
