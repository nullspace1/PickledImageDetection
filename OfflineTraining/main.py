import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.Hypernetwork import HyperNetwork
from Model.ImageProcessor import ImageProcessor
from Model.TemplateProcessor import TemplateProcessor
from OfflineTraining.DataLoader import DataLoader
from OfflineTraining.DataCreator import DataCreator
from OfflineTraining.Trainer import OfflineTrainer

import argparse
from multiprocessing import freeze_support

def main():

    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--samples_train', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--samples_validation', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--model_path', type=str, default="data/model.pth", help='Path to the model')
    parser.add_argument('--generated_data_path', type=str, default="data/generated_data", help='Path to the generated data')
    parser.add_argument('--screenshots_path', type=str, default="data/screenshots", help='Path to the screenshots')
    parser.add_argument('--training_data_path', type=str, default="data/training_data.npy", help='Path to the training data')
    parser.add_argument('--validation_data_path', type=str, default="data/validation_data.npy", help='Path to the validation data')
    parser.add_argument('--logging_interval', type=int, default=2, help='Interval to log the progress')

    args = parser.parse_args()

    image_processor = ImageProcessor()
    template_processor = TemplateProcessor(1000)
    hypernetwork = HyperNetwork(image_processor, template_processor)

    data_creator_train = DataCreator(args.screenshots_path,  args.generated_data_path, templates_per_screenshot=1, samples=args.samples_train)
    data_creator_validation = DataCreator(args.screenshots_path, args.generated_data_path, templates_per_screenshot=1, samples=args.samples_validation)
    training_data_loader = DataLoader(args.training_data_path, data_creator_train)
    validation_data_loader = DataLoader(args.validation_data_path, data_creator_validation)

    trainer = OfflineTrainer(hypernetwork, training_data_loader, validation_data_loader, torch.optim.Adam(hypernetwork.parameters(), lr=0.001), args.model_path, args.logging_interval)

    trainer.train(args.epochs)

if __name__ == "__main__":
    freeze_support()
    main()






