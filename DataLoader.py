import os
import numpy as np
import torch
import cv2
import random
from DataCreator import DataCreator
import time

class Cache:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.cache = {}
        
    def add_to_cache(self, data, start_index):
        for i in range(start_index, start_index + self.cache_size):
            screenshot_file_link, heatmap_file_link, template_file_link  = data[i]
            screenshot = torch.load(screenshot_file_link)
            heatmap = torch.load(heatmap_file_link)
            template = torch.load(template_file_link)
            self.cache[i] = (screenshot, template, heatmap)
        
    def init_cache(self, data):
       self.add_to_cache(data, 0)
    
    def is_in_cache(self, index):
        return index in self.cache
    
    def update_cache(self, data, index):
        self.add_to_cache(data, index)

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data_save_path, data_creator,cache, batch_size=1):
        self.data_save_path = data_save_path
        self.data_creator = data_creator
        self.batch_size = batch_size
        self.load_data()
        self.cache = cache


    def load_data(self):
        if not os.path.exists(self.data_save_path):
            self.data_creator.create_data(self.data_save_path)
        self.data = np.load(self.data_save_path, allow_pickle=True)
        self.cache.init_cache(self.data)


    def __len__(self):
        return len(self.data)
    
    def is_in_cache(self, index):
        return index in self.cache
    
    def __getitem__(self, index):
        
        
        if (self.is_in_cache(index)):
            return self.cache[index]
        else:
            self.cache.update_cache(self.data, index)
            screenshot_file_link, heatmap_file_link, template_file_link  = self.data[index]
            screenshot = torch.load(screenshot_file_link)
            heatmap = torch.load(heatmap_file_link)
            template = torch.load(template_file_link)
            return screenshot,  template, heatmap
        
    
