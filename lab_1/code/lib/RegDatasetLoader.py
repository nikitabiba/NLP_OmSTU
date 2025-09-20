import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class RegDatasetLoader(Dataset):
    def __init__(self, path, count, begin):
        df = pd.read_csv(path, sep=';')

        if begin == "head":
            orig_X = df.drop(['quality'], axis=1)
            X = orig_X.head(int(len(orig_X)*count)).values
            orig_y = df['quality']
            y = orig_y.head(int(len(orig_y)*count)).values
        elif begin == "tail":
            orig_X = df.drop(['quality'], axis=1)
            X = orig_X.tail(int(len(orig_X)*count)).values
            orig_y = df['quality']
            y = orig_y.tail(int(len(orig_y)*count)).values

        self.length = len(y)

        ss = StandardScaler()
        ss.fit(X)
        X = ss.fit_transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.X[index], self.y[index]