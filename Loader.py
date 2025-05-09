import torch

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        image = image[:,:,0]
        label = label[:,:,0]
        image = image[:,:,None]
        label = label[:,:,None]
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return (image, label)

