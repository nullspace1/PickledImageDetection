import torch
from Hypernetwork import HyperNetwork
from ImageProcessor import ImageProcessor
from TemplateProcessor import TemplateProcessor
from DataLoader import DataLoader
from DataCreator import DataCreator
from Trainer import Trainer
from DataLoader import Cache

import argparse

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--model_path', type=str, default="data/model.pth", help='Path to the model')
parser.add_argument('--cache_size', type=int, default=1000, help='Cache size for the data loader')

args = parser.parse_args()

image_processor = ImageProcessor()
template_processor = TemplateProcessor(1000)
hypernetwork = HyperNetwork(image_processor, template_processor)

data_creator = DataCreator("data/screenshots", "data/templates", templates_per_screenshot=10, samples=args.samples)
training_data_loader = DataLoader("data/training_data.npy", data_creator, Cache(args.cache_size), batch_size=args.batch_size)
validation_data_loader = DataLoader("data/validation_data.npy", data_creator, Cache(args.cache_size), batch_size=args.batch_size)

trainer = Trainer(hypernetwork, training_data_loader, torch.optim.Adam(hypernetwork.parameters(), lr=0.001), args.model_path)

trainer.train(args.epochs)









