import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class Trainer():
    def __init__(self, model, dataloader, optimizer : torch.optim.Optimizer, model_path = None):
        super(Trainer, self).__init__()
        self.model = model
        if model_path is not None and isinstance(model_path, str) and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.optimizer = optimizer
        self.data = dataloader
        self.best_loss = float('inf')
        
    def train_epoch(self,epoch):
        running_loss = 0.0
        pbar = tqdm(self.data, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            images, templates, targets = batch
            
            self.optimizer.zero_grad()
            outputs = self.model(images, templates)
            loss = self.model.loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss/len(self.data):.3f}'})
                

        if running_loss < self.best_loss:
            self.best_loss = running_loss
            torch.save(self.model.state_dict(), f'best_model.pth')
            
    def train(self,epochs):
        for epoch in range(epochs):
            self.train_epoch(epoch)
            self.validate(epoch)
        
        
    def validate(self,epoch):
        self.model.eval()
        running_loss = 0.0
        pbar = tqdm(self.data, desc=f'Validation')
        for _, batch in enumerate(pbar):
            images, templates, targets = batch  
            outputs = self.model(images, templates)
            loss = self.model.loss(outputs, targets)    
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss/len(self.data):.3f}'})    
        print(f'Validation loss: {running_loss / len(self.data):.3f}')    
        
    
