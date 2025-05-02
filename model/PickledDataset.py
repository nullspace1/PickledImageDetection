import torch
from torch.utils.data import Dataset
import numpy as np

class PickledDataset(Dataset):
    def __init__(self, data):
        self.data = np.load(data, allow_pickle=True)   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image,crop,rect,found = self.data[idx][0],self.data[idx][1],self.data[idx][2],self.data[idx][3]
        image = torch.tensor(image, dtype=torch.float32)
        crop = torch.tensor(crop, dtype=torch.float32)
        rect = torch.tensor(rect, dtype=torch.float32)
        found = torch.tensor(found, dtype=torch.float32)
        return (image, crop), (rect, found)