import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]
