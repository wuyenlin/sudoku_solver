#!/usr/bin/python3

import torch
import torch.nn as nn
from torchvision import models

class tell_digit(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.Dropout(p=0.8, inplace=False),
            nn.Linear(128, 10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        return self.fc(x)


if __name__ == "__main__":
    model = ra_net()
    inp = torch.rand(1,3,256,256)
    output = model(inp)
    print(output.shape)