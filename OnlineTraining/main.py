import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from Model.Hypernetwork import HyperNetwork
from OnlineTraining.OnlineTrainer import OnlineTrainer
from Model.ImageProcessor import ImageProcessor
from Model.TemplateProcessor import TemplateProcessor
from OnlineTraining.DataProvider import DataProvider
import argparse

args = argparse.ArgumentParser()

args.add_argument("--port", type=int, default=5000)
args.add_argument("--host", type=str, default="0.0.0.0")
args.add_argument("--checkpoint_interval", type=int, default=10)
args.add_argument("--max_iterations", type=int, default=100000)
args.add_argument("--model_folder_path", type=str, default="models")
args.add_argument("--data_variant_to_generate", type=int, default=10)
args.add_argument("--max_data", type=int, default=500)
args = args.parse_args()

image_processor = ImageProcessor()
template_processor = TemplateProcessor(1000)
model = HyperNetwork(image_processor, template_processor)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

online_trainer = OnlineTrainer(model, optimizer, DataProvider(args.host, args.port, args.data_variant_to_generate, args.max_data), args.model_folder_path, args.checkpoint_interval, args.max_iterations)
online_trainer.listen()