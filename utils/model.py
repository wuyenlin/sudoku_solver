#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk1= nn.Sequential(
            nn.Conv2d(1, 28, 5, 1),
            nn.Conv2d(28, 28, 5, 1, bias=False),
            nn.BatchNorm2d(28)
        )

        self.blk2 = nn.Sequential(
            nn.Conv2d(28, 56, 3, 1),
            nn.Conv2d(56, 56, 3, 1, bias=False),
            nn.BatchNorm2d(56)
        )

        self.pool = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(504, 84, bias=False)
        self.fc2 = nn.Linear(84, 10, bias=False)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x):
        x = self.pool(self.blk1(x))
        x = self.dropout(x)
        x = self.pool(self.blk2(x))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)

        return output
