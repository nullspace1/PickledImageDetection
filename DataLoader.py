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
        
        screenshot = cv2.imread(screenshot_file_link)
        heatmap = cv2.imread(heatmap_file_link)
        template = cv2.imread(template_file_link)
        
        screenshot = torch.from_numpy(screenshot).permute(2, 0, 1)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1)
        template = torch.from_numpy(template).permute(2, 0, 1)
        
        # make the heatmap single channel
        heatmap = heatmap.float().mean(dim=0).unsqueeze(0) / 255
        
        return screenshot.float()/255,  template.float()/255, heatmap
        
if __name__ == "__main__":
    data_loader = DataLoader("data/training_data.npy", DataCreator("data/screenshots", "data/templates"))
    print(len(data_loader))
    print(data_loader[0])
    
