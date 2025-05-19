import torch
from tqdm import tqdm
import os
import logging
import traceback
import matplotlib.pyplot as plt
import numpy as np

from Hypernetwork import HyperNetwork
from ImageProcessor import ImageProcessor
from DataLoader import DataLoader
from DataCreator import DataCreator
from TemplateProcessor import TemplateProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/training.log'),
        logging.StreamHandler()
    ]
)

class Trainer():
    def __init__(self, model, dataloader, optimizer : torch.optim.Optimizer, model_path):
        super(Trainer, self).__init__()
        self.model = model
        self.model_path = model_path
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.optimizer = optimizer
        self.data = dataloader
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self,epoch):
        running_loss = 0.0
        pbar = tqdm(self.data, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            try:
                images, templates, heatmaps = batch
                
                self.optimizer.zero_grad()
                outputs = self.model(images, templates)
                loss = self.model.loss(outputs, heatmaps)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': f'{running_loss/len(self.data):.3f}'})
                
                del images, templates, heatmaps
                
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
            torch.save(self.model.state_dict(), self.model_path)
            
    def train(self,epochs):
        for epoch in range(epochs):
            self.train_epoch(epoch)
            self.validate(epoch)
            self.plot_losses()
        
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
        print(f'Validation loss: {val_loss:.3f}')
        
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
        
    def test(self):
        screenshot, template = torch.randn(1, 3, 1080, 1920), torch.randn(1, 3, 100,100)
        output = self.model(screenshot, template)
        
if __name__ == "__main__":
    model = HyperNetwork(
            image_processor=ImageProcessor(),
            template_processor=TemplateProcessor(1000))
    trainer = Trainer(
        model=model,
        dataloader=DataLoader("data/training_data.npy", DataCreator("data/screenshots", "data/templates")),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        model_path="data/model.pth"
    )
    trainer.test()
        
    
