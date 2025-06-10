import torch
from Model.Hypernetwork import HyperNetwork
import time
import socket,struct
import numpy as np
from OnlineTraining.DataProvider import DataProvider
import os
import cv2

class OnlineTrainer():
    
    def __init__(self, model : HyperNetwork, optimizer : torch.optim.Optimizer, data_provider : DataProvider, model_folder_path : str = "models", save_interval : int = 10, max_iterations : int = 100000):
        super(OnlineTrainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.data_provider = data_provider
        self.save_interval = save_interval
        self.MAX_ITERATIONS = max_iterations
        self.loss_history = []
        self.model_folder_path = model_folder_path
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)
            os.makedirs(f"{self.model_folder_path}/sample")

    def train(self, image, template, heatmap):
       
        print(f"Training with data: {image.shape}, {template.shape}, {heatmap.shape}")
       
        result = self.model.forward(image, template)
        loss = self.model.loss(result, heatmap)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())
        print(f"Loss: {loss.item()}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def save_progress(self, path):
        print(f"Saving progress to {path}")
        torch.save(self.model.state_dict(), path)
        
    def get_new_data(self):
        screenshot, template, rectangle = self.data_provider.get_next_data()
        
        heatmap = np.zeros((1, 1, screenshot.shape[0], screenshot.shape[1]))
        heatmap[0, 0, rectangle[1]:rectangle[1]+rectangle[3], rectangle[0]:rectangle[0]+rectangle[2]] = 1
        
        screenshot = torch.from_numpy(screenshot).float().permute(2, 0, 1) / 255
        template = torch.from_numpy(template).float().permute(2, 0, 1) / 255
        heatmap = torch.from_numpy(heatmap).float().mean(dim=0)
        
        print(f"Received data: {screenshot.shape}, {template.shape}, {heatmap.shape}")

        
        return screenshot, template, heatmap
        

    def log_progress(self, i):
        print(f"Iteration {i} - Loss: {self.loss_history[-1]}")
    
    def listen(self):
        self.data_provider.start_gathering()
        for i in range(self.MAX_ITERATIONS):
            screenshot, template, heatmap = self.get_new_data()
            self.train(screenshot, template, heatmap)
            if i % self.save_interval == 0:
                self.save_progress(f"{self.model_folder_path}/progress_{self.model.hash()}.pth")
                self.log_progress(i)
        