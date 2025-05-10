import torch
import numpy as np

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        """
        1. Read the image and the label
        2. Convert the image and the label to grayscale, by choosing the first channel -> all rows, all columns, first channel (grayscale)
        3. Add a channel dimension -> all rows, all columns, all channels (RGB)
        4. Convert the image and the label to torch tensors
        5. Return the image and the label
        """
        image, label = self.data[index]
        image = image[:,:,0]
        label = label[:,:,0]
        image = image[:,:,None]
        label = label[:,:,None]
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return (image, label)

