import torch
from tqdm import tqdm
import os
import logging
import traceback
import matplotlib.pyplot as plt
import numpy as np
import time
import psutil
import gc
from Model.Hypernetwork import HyperNetwork
from Model.ImageProcessor import ImageProcessor
from OfflineTraining.DataLoader import DataLoader
from OfflineTraining.DataCreator import DataCreator
from Model.TemplateProcessor import TemplateProcessor
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/training.log'),
        logging.StreamHandler()
    ]
)

class OfflineTrainer():
    def __init__(self, model, dataloader, optimizer : torch.optim.Optimizer, model_path, patience = 10):
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
        self.data = dataloader
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.patience = patience
        self.best_loss = float('inf')
        self.best_loss_epoch = 0
        
    def is_compatible(self, model_path):
        return os.path.exists(model_path) and model_path[:-4].endswith(self.model.hash()) and model_path[-4:] == ".pth"

    def get_memory_stats(self):
        # Get GPU memory stats if CUDA is available
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        
        # Get CPU memory stats
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**2  # Convert to MB
        
        return gpu_memory, cpu_memory

    def log_memory_stats(self):
        gpu_memory, cpu_memory = self.get_memory_stats()
        self.memory_stats['gpu_memory'].append(gpu_memory)
        self.memory_stats['cpu_memory'].append(cpu_memory)
        self.memory_stats['timestamps'].append(time.time())

    def plot_memory_usage(self):
        plt.figure(figsize=(12, 6))
        
        # Convert timestamps to relative time in minutes
        timestamps = np.array(self.memory_stats['timestamps'])
        relative_time = (timestamps - timestamps[0]) / 60
        
        plt.plot(relative_time, self.memory_stats['gpu_memory'], label='GPU Memory (MB)', color='blue')
        plt.plot(relative_time, self.memory_stats['cpu_memory'], label='CPU Memory (MB)', color='red')
        
        plt.xlabel('Time (minutes)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('data/memory_usage.png')
        plt.close()

    def train_epoch(self,epoch):
        running_loss = 0.0
        pbar = tqdm(self.data, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Log memory stats at the start of each batch
                self.log_memory_stats()
                
                images, templates, heatmaps = batch
                
                self.optimizer.zero_grad()
                start_time = time.time()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(images, templates)
                logging.info(f"Forward pass took {time.time() - start_time:.3f} seconds")
                       
                print(f"Outputs shape: {outputs.shape}, Heatmaps shape: {heatmaps.shape}")
                if outputs.shape[1] != heatmaps.shape[1] or outputs.shape[2] != heatmaps.shape[2]:
                    heatmaps = F.interpolate(heatmaps, size=(outputs.shape[1], outputs.shape[2]), mode='bilinear', align_corners=False)
                
                loss = self.model.loss(outputs, heatmaps)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': f'{running_loss/len(self.data):.3f}'})
                
                # Clean up memory
                del images, templates, heatmaps, outputs
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error in training batch: {str(e)}")
                logging.error(f"Batch shapes - Images: {images.shape if 'images' in locals() else 'N/A'}, "
                            f"Templates: {templates.shape if 'templates' in locals() else 'N/A'}, "
                            f"Heatmaps: {heatmaps.shape if 'heatmaps' in locals() else 'N/A'}")
                logging.error(traceback.format_exc())
                continue

        epoch_loss = running_loss / len(self.data)
        self.train_losses.append(epoch_loss)
        
        if running_loss < self.best_loss:
            self.best_loss = running_loss
            self.best_loss_epoch = epoch
            torch.save(self.model.state_dict(), self.model_path)
            
    def train(self,epochs):
        for epoch in range(epochs):
            self.train_epoch(epoch)
            self.validate(epoch)
            self.plot_losses()
            self.plot_memory_usage()  # Plot memory usage after each epoch
            if epoch - self.best_loss_epoch > self.patience:
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break
        
    def validate(self,epoch):
        self.model.eval()
        running_loss = 0.0
        pbar = tqdm(self.data, desc=f'Validation')
        for _, batch in enumerate(pbar):
            images, templates, heatmaps = batch  
            outputs = self.model(images, templates)
            loss = self.model.loss(outputs, heatmaps)    
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss/len(self.data):.3f}'})    
        val_loss = running_loss / len(self.data)
        self.val_losses.append(val_loss)
        print(f'Validation loss at epoch {epoch} : {val_loss:.3f}')
        
    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig('data/loss_plot.png')
        plt.close()
        
