import torch
from torch.utils.data.dataset import Dataset

class ECGDataset(Dataset):
    def __init__(self, data, labels = None) -> None:
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = torch.tensor(self.data[index]).float()
        
        if self.labels is not None:
            y = torch.tensor(self.labels[index])
            return x, y
        else:
            return x

    def __len__(self):
        return self.data.shape[0]