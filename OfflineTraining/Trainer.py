import torch
from tqdm import tqdm
import os
import logging
import traceback
import matplotlib.pyplot as plt
import numpy as np
import time
import psutil
from Model.Hypernetwork import HyperNetwork
from Model.ImageProcessor import ImageProcessor
from OfflineTraining.DataLoader import DataLoader
from OfflineTraining.DataCreator import DataCreator
from Model.TemplateProcessor import TemplateProcessor
import torch.nn.functional as F
import cv2
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/training.log'),
        logging.StreamHandler()
    ]
)

class OfflineTrainer():
    def __init__(self, model, training_data_loader,validation_data_loader, optimizer : torch.optim.Optimizer, model_path, logging_interval = 10, patience = 10):
        
        super(OfflineTrainer, self).__init__()
        
        self.model = model
        self.model_path = model_path[:-4] + "_" + self.model.hash() + ".pth"
        self.memory_stats = {
            'gpu_memory': [],
            'cpu_memory': [],
            'timestamps': []
        }

        if self.is_compatible(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            logging.info(f"No model found at {self.model_path}, creating new model")
            
        self.optimizer = optimizer
        self.validation_data = validation_data_loader
        self.training_data = training_data_loader
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.patience = patience
        self.best_loss = float('inf')
        self.best_loss_epoch = 0
        self.logging_interval = logging_interval
        
        self.results_path = 'data/results'
        os.makedirs(self.results_path, exist_ok=True)
        
        
    def is_compatible(self, model_path):
        return os.path.exists(model_path) and model_path[:-4].endswith(self.model.hash()) and model_path[-4:] == ".pth"

    def get_memory_stats(self):
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
        
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**2
        
        return gpu_memory, cpu_memory

    def log_memory_stats(self):
        gpu_memory, cpu_memory = self.get_memory_stats()
        self.memory_stats['gpu_memory'].append(gpu_memory)
        self.memory_stats['cpu_memory'].append(cpu_memory)
        self.memory_stats['timestamps'].append(time.time())

    def plot_memory_usage(self):
        plt.figure(figsize=(12, 6))
        
        timestamps = np.array(self.memory_stats['timestamps'])
        relative_time = (timestamps - timestamps[0]) / 60
        
        plt.plot(relative_time, self.memory_stats['gpu_memory'], label='GPU Memory (MB)', color='blue')
        plt.plot(relative_time, self.memory_stats['cpu_memory'], label='CPU Memory (MB)', color='red')
        
        plt.xlabel('Time (minutes)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.results_path}/memory_usage.png')
        plt.close()

    def train_epoch(self,epoch):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.training_data, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                self.log_memory_stats()
                
                images, templates, heatmaps = batch
                
                outputs = self.model(images, templates) 
                loss = self.model.loss(outputs, heatmaps)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.3f}', 'avg_loss': f'{running_loss/(batch_idx+1):.3f}'})

                del images, templates, heatmaps, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Error in training batch: {str(e)}")
                logging.error(f"Batch shapes - Images: {images.shape if 'images' in locals() else 'N/A'}, "
                            f"Templates: {templates.shape if 'templates' in locals() else 'N/A'}, "
                            f"Heatmaps: {heatmaps.shape if 'heatmaps' in locals() else 'N/A'}")
                logging.error(traceback.format_exc())
                continue

        epoch_loss = running_loss / len(self.training_data)
        self.train_losses.append(epoch_loss)
        
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_loss_epoch = epoch
            torch.save(self.model.state_dict(), self.model_path)
            
    def train(self,epochs):
        patience_counter = 0
        for epoch in range(epochs):
            self.train_epoch(epoch)
            self.validate(epoch)
            
            # Early stopping logic
            if len(self.val_losses) > 0:
                current_val_loss = self.val_losses[-1]
                if current_val_loss < self.best_loss:
                    self.best_loss = current_val_loss
                    self.best_loss_epoch = epoch
                    patience_counter = 0
                    torch.save(self.model.state_dict(), self.model_path)
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            if epoch % self.logging_interval == 0:
                image, template, heatmap, outputs = self.sample_output
                cv2.imwrite(f'{self.results_path}/outputs.png', outputs.permute(1, 2, 0).cpu().detach().numpy() * 255)
                cv2.imwrite(f'{self.results_path}/heatmaps.png', heatmap.permute(1, 2, 0).cpu().detach().numpy() * 255)
                cv2.imwrite(f'{self.results_path}/screenshots.png', image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255)
                cv2.imwrite(f'{self.results_path}/templates.png', template.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255)
            self.plot_losses()
            self.plot_memory_usage()  # Plot memory usage after each epoch
        
    def validate(self,epoch):
        self.model.eval()
        running_loss = 0.0
        pbar = tqdm(self.validation_data, desc=f'Validation')
        val_length = len(self.validation_data)
        rand_pos = random.randint(0, max(0, val_length - 1)) if val_length > 0 else 0
        for i, batch in enumerate(pbar):
            image, template, heatmap = batch  
            outputs = self.model(image, template)
            loss = self.model.loss(outputs, heatmap)    
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'avg_loss': f'{running_loss/(i+1):.3f}'})
            
            if i == rand_pos and epoch % self.logging_interval == 0:
                self.sample_output = image, template, heatmap, outputs
           
        
        val_loss = running_loss / len(self.validation_data)
        self.val_losses.append(val_loss)
        
    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.results_path}/loss_plot.png')
        plt.close()
        
