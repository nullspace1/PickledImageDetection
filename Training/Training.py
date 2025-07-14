import time
import torch
from Model.Hypernetwork import HyperNetwork
from Training import Recorder
from Training.DataCreator import DataCreator
from Training.Recorder import EpochLog
from Training.Recorder import Recorder
from tqdm import tqdm

class Trainer:
    
    def __init__(self, model : HyperNetwork, data_creator_train : DataCreator, data_creator_validate : DataCreator, optimizer : torch.optim.Optimizer, epochs : int, recorder : Recorder, patience : int = 10):
        self.model = model
        self.optimizer = optimizer
        self.data_creator_train = data_creator_train
        self.data_creator_validate = data_creator_validate
        self.epochs = epochs
        self.recorder = recorder
        self.patience = patience
        
    def train(self):
        
        for epoch in range(self.epochs):
            
            print(f"Training epoch {epoch}")
            
            training_loss = self.train_epoch()
            validation_loss = self.validate()
            self.recorder.record(self.model, EpochLog(epoch, training_loss, validation_loss))
            
            if epoch - self.recorder.best_epoch > self.patience:
                break
            
        
        
    def train_epoch(self) -> list[float]:
        
        losses = []

        for i in tqdm(range(self.data_creator_train.sample_size()), desc="Training batch"):
            sample_image, sample_template, sample_heatmap = self.data_creator_train.create_data(i)
            output_heatmap = self.model(sample_image, sample_template)
            output_heatmap = output_heatmap.squeeze(0).squeeze(0)
            loss = self.model.loss(output_heatmap, sample_heatmap)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            self.recorder.visualize(sample_image, sample_heatmap, output_heatmap)
            
        return losses
        
        
        
    def validate(self) -> list[float]:
        
        losses = []
        
        for i in tqdm(range(self.data_creator_validate.sample_size()), desc="Validating batch"):
            sample_image, sample_template, sample_heatmap = self.data_creator_validate.create_data(i)
            output_heatmap = self.model(sample_image, sample_template)
            output_heatmap = output_heatmap.squeeze(0).squeeze(0)
            loss = self.model.loss(output_heatmap, sample_heatmap)
            losses.append(loss.item())
            
        return losses