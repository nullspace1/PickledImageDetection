from Builder import DataBuilder
from Loader import DataLoader
from Model import Model
from Trainer import Trainer
import torch
import numpy as np
import os
model_path = "data/model.pth"

builder = DataBuilder()

builder.build_data("data/screenshots", "data/training_data.npy")
builder.build_data("data/screenshots", "data/validation_data.npy")

training_data = np.load("data/training_data.npy", allow_pickle=True)
validation_data = np.load("data/validation_data.npy", allow_pickle=True)

training_loader = DataLoader(training_data)
validation_loader = DataLoader(validation_data)

model = Model()

if os.path.exists(model_path):
    model.load(model_path)


trainer = Trainer(model, training_loader, validation_loader)

trainer.train(5,save_path=model_path)