from model.capturer import Capturer
import numpy as np
from model.Model import Model
from model.Loss import Loss
from model.PickledDataset import PickledDataset
import torch
from torch.utils.data import DataLoader

capturer = Capturer(0.3,0.3,0.8,1.2,0.8,1.2,10)
capturer.load_training_data("data/screenshots", "data/training_data.npy")

dataset = PickledDataset("data/training_data.npy")

model = Model()
loss_func = Loss(0.5,0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loader = DataLoader(dataset, batch_size=3, shuffle=True)

for epoch in range(10):
    running_loss = 0.0
    print(f"Epoch {epoch}")
    for (image,crop), (box_gt, found_gt) in loader:
        optimizer.zero_grad()
        box, found = model(image, crop)
        loss = loss_func.forward(box, found, box_gt, found_gt)
        print('loss: ',loss)
        running_loss += loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch} loss: {running_loss/len(loader)}")

torch.save(model.state_dict(), "model.pt")
        
        
