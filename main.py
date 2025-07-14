import torch
from Training.Training import Trainer
from Training.DataCreator import BasicShapeDataCreator
from Training.Recorder import Recorder
from Model.Hypernetwork import HyperNetwork
from Model.ImageProcessor import ImageProcessor
from Model.TemplateProcessor import TemplateProcessor

image_processor = ImageProcessor(1000, 64, 16)
template_processor = TemplateProcessor(1000)

model = HyperNetwork(image_processor, template_processor)

weight_path = f"models/model_{model.hash()}.pth"

data_creator_train = BasicShapeDataCreator(64)
data_creator_validate = BasicShapeDataCreator(64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

recorder = Recorder(weight_path, "recordings")

trainer = Trainer(model, data_creator_train, data_creator_validate, optimizer, 100, recorder)

trainer.train()