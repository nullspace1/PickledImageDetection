import os
import numpy as np
import torch
import cv2
import random
from DataCreator import DataCreator

class DataLoader(torch.utils.data.Dataset):
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
        screenshot, template, box = self.data[index]
        screenshot = torch.from_numpy(screenshot).permute(2, 0, 1)
        template = torch.from_numpy(template).permute(2, 0, 1)
        box = torch.from_numpy(np.array(box)).unsqueeze(0)
        return screenshot.float(), template.float(), box.float()
        
if __name__ == "__main__":
    data_loader = DataLoader("data/training_data.npy", DataCreator("data/screenshots", "data/templates"))
    print(len(data_loader))
    print(data_loader[0])
    
