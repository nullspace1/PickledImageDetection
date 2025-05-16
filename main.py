import torch
from Hypernetwork import HyperNetwork
from ImageProcessor import ImageProcessor
from TemplateProcessor import TemplateProcessor
from DataLoader import DataLoader
from DataCreator import DataCreator
from Trainer import Trainer

BATCH_SIZE = 1
SAMPLES = 1
EPOCHS = 1


## Initialize the model
image_processor = ImageProcessor()
template_processor = TemplateProcessor(1000)
hypernetwork = HyperNetwork(image_processor, template_processor)


## Initialize the data
data_creator = DataCreator("data/screenshots", "data/templates",batch_size=BATCH_SIZE,samples=SAMPLES)
training_data_loader = DataLoader("data/training_data.npy", data_creator)
validation_data_loader = DataLoader("data/validation_data.npy", data_creator)


## Initialize the trainer
trainer = Trainer(hypernetwork, training_data_loader, torch.optim.Adam(hypernetwork.parameters(), lr=0.001), "data/model.pth")


## Train the model
trainer.train(EPOCHS)









