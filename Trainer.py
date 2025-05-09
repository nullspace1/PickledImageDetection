import torch
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(self, model, train_loader, val_loader=None, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.model.to(self.device)
        

        self.scheduler = ReduceLROnPlateau(
            self.model.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.model.optimizer.zero_grad()
            x_features = self.model(x)
            y_features = self.model(y)
            loss = self.model.loss(x_features, y_features)
            
            loss.backward()
            self.model.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'lr': self.model.optimizer.param_groups[0]['lr']
            })
        
        return total_loss / num_batches

    def validate(self):
        if self.val_loader is None:
            return None
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                

                x_features = self.model(x)
                y_features = self.model(y)
                loss = self.model.loss(x_features, y_features)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': total_loss / num_batches})
        
        return total_loss / num_batches

    def train(self, epochs, patience=10, save_path='best_model.pth'):
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {self.device}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['learning_rates'].append(self.model.optimizer.param_groups[0]['lr'])
            
            if self.val_loader is not None:
                val_loss = self.validate()
                self.history['val_loss'].append(val_loss)
                

                if val_loss is not None:
                    self.scheduler.step(val_loss)
                
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model(save_path)
                    print(f"New best model saved! Validation loss: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        break
                
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f}")
        
        return self.history

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'history': self.history
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)     

