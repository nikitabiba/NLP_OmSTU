import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, path, n, count, begin):
        df = pd.read_csv(path, sep=';')[:n]
        if begin == "head":
            orig_X = df.drop(['category'], axis=1)
            X = orig_X.head(int(len(orig_X)*count)).values
            orig_y = df['category']
            y = orig_y.head(int(len(orig_y)*count)).values
        elif begin == "tail":
            orig_X = df.drop(['category'], axis=1)
            X = orig_X.tail(int(len(orig_X)*count)).values
            orig_y = df['category']
            y = orig_y.tail(int(len(orig_y)*count)).values
        self.length = len(y)
        self.X = X
        self.y = y

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.X[index][0], self.y[index]