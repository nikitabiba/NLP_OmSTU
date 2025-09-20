import torch
import torch.nn as nn


class ClassFCNN(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.layer1 = nn.Linear(inp, 64)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 128)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 256)
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(256, out)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.act3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        return x