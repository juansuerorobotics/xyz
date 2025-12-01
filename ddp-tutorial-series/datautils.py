
import torch
from torch.utils.data import Dataset

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]


    
class ControlledDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

        # True underlying rule: only feature 0 and 5 matter
        self.W_true = torch.zeros(20)
        self.W_true[0] = 2.0   # brightness
        self.W_true[5] = -1.0  # speed
        self.b_true = 0.5

        self.data = []
        for _ in range(size):
            x = torch.rand(20)  # inputs in [0,1]
            # compute scalar y using our controlled rule
            y = (self.W_true * x).sum() + self.b_true
            # optional noise
            #y = y + 0.01 * torch.randn(())
            # store as (20,) and (1,)
            self.data.append((x, y.unsqueeze(0)))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
    
    