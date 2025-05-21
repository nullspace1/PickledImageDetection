import torch
from Hypernetwork import HyperNetwork
from ImageProcessor import ImageProcessor
from TemplateProcessor import TemplateProcessor
from DataLoader import DataLoader
from DataCreator import DataCreator
from Trainer import Trainer

import argparse

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--samples_train', type=int, default=1000, help='Number of samples to generate')
parser.add_argument('--samples_validation', type=int, default=100, help='Number of samples to generate')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--model_path', type=str, default="data/model.pth", help='Path to the model')
parser.add_argument('--cache_size', type=int, default=1000, help='Cache size for the data loader')
parser.add_argument('--generated_data_path', type=str, default="data/generated_data", help='Path to the generated data')
parser.add_argument('--screenshots_path', type=str, default="data/screenshots", help='Path to the screenshots')
parser.add_argument('--templates_path', type=str, default="data/templates", help='Path to the templates')
parser.add_argument('--training_data_path', type=str, default="data/training_data.npy", help='Path to the training data')
parser.add_argument('--validation_data_path', type=str, default="data/validation_data.npy", help='Path to the validation data')

args = parser.parse_args()

image_processor = ImageProcessor()
template_processor = TemplateProcessor(1000)
hypernetwork = HyperNetwork(image_processor, template_processor)

data_creator_train = DataCreator(args.screenshots_path, args.templates_path, args.generated_data_path, templates_per_screenshot=10, samples=args.samples_train)
data_creator_validation = DataCreator(args.screenshots_path, args.templates_path, args.generated_data_path, templates_per_screenshot=10, samples=args.samples_validation)
training_data_loader = DataLoader(args.training_data_path, data_creator_train, batch_size=args.batch_size)
validation_data_loader = DataLoader(args.validation_data_path, data_creator_validation, batch_size=args.batch_size)

trainer = Trainer(hypernetwork, training_data_loader, torch.optim.Adam(hypernetwork.parameters(), lr=0.001), args.model_path)

trainer.train(args.epochs)









