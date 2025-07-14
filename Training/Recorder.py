import time
import torch
from Model.Hypernetwork import HyperNetwork
import matplotlib.pyplot as plt
import cv2

class EpochLog:
    
    def __init__(self, epoch : int, training_loss : list[float], validation_loss : list[float]):
        self.epoch = epoch
        self.training_loss = training_loss
        self.validation_loss = validation_loss
        
    def total_validation_loss(self) -> float:
        return sum(self.validation_loss) / len(self.validation_loss)
    
    def total_training_loss(self) -> float:
        return sum(self.training_loss) / len(self.training_loss)
    
        
class Recorder:
    
    def __init__(self, model_path : str, log_path : str):
        self.model_path : str = model_path
        self.log_path : str = log_path
        self.best_loss : float = float('inf')
        self.best_epoch : int = 0
        self.logs : list[EpochLog] = []
           
    def save_model(self, model : HyperNetwork, epoch : int) -> None:
        torch.save(model.state_dict(), self.model_path)
        
    def record(self, model : HyperNetwork, epoch_log : EpochLog) -> None:
        
        print(f"Epoch {epoch_log.epoch} validation loss: {epoch_log.total_validation_loss()}")
        
        if epoch_log.total_validation_loss() < self.best_loss:
            self.best_loss = epoch_log.total_validation_loss()  
            self.best_epoch = epoch_log.epoch
            print(f"New best validation loss: {self.best_loss} at epoch {self.best_epoch}")
            self.save_model(model, epoch_log.epoch)
            
            
    def plot_logs(self) -> None:
        
        validation_losses = [log.total_validation_loss() for log in self.logs]
        training_losses = [log.total_training_loss() for log in self.logs]
        
        plt.plot(validation_losses, label="Validation Loss")
        plt.plot(training_losses, label="Training Loss")
        plt.legend()
        plt.savefig(self.log_path)
        
    def visualize(self, sample_image : torch.Tensor, sample_heatmap : torch.Tensor, output_heatmap : torch.Tensor) -> None:
        # Convert tensors to numpy arrays and scale to 0-255 range
        sample_image_np = (sample_image.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype('uint8')
        sample_heatmap_np = (sample_heatmap.squeeze(0).detach().numpy() * 255).astype('uint8') 
        output_heatmap_np = (output_heatmap.squeeze(0).detach().numpy() * 255).astype('uint8')
        
        # Save images using cv2
        cv2.imwrite(self.log_path + "/sample_image.png", cv2.cvtColor(sample_image_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(self.log_path + "/sample_heatmap.png", sample_heatmap_np)
        cv2.imwrite(self.log_path + "/output_heatmap.png", output_heatmap_np)
        
        