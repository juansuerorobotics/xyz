
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

        # Strong, obvious rule:
        self.W_true = torch.zeros(20)
        self.W_true[0] = 10.0   # feature 0: big positive
        self.W_true[5] = -5.0   # feature 5: big negative
        self.b_true = 2.0

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
    
    