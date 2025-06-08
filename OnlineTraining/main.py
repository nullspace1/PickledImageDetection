import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from Model.Hypernetwork import HyperNetwork
from OnlineTraining.OnlineTrainer import OnlineTrainer
from Model.ImageProcessor import ImageProcessor
from Model.TemplateProcessor import TemplateProcessor
import time
import threading
import socket
import struct
from OnlineTraining.DataProvider import DataProvider
from OnlineTraining.DataReceiver import DataReceiver
import argparse

args = argparse.ArgumentParser()

args.add_argument("--port", type=int, default=5000)
args.add_argument("--host", type=str, default="0.0.0.0")
args.add_argument("--save_interval", type=int, default=10)
args.add_argument("--max_iterations", type=int, default=100000)
args.add_argument("--model_folder_path", type=str, default="models")
args = args.parse_args()

image_processor = ImageProcessor()
template_processor = TemplateProcessor(1000)
model = HyperNetwork(image_processor, template_processor)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

online_trainer = OnlineTrainer(model, optimizer, DataProvider(DataReceiver(args.port, args.host), 0.3, 200), args.model_folder_path, args.save_interval, args.max_iterations)
online_trainer.listen()