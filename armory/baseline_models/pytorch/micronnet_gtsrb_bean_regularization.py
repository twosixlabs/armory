from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nclasses = 43  # GTSRB has 43 classes


class Micronnet(nn.Module):
    def __init__(self) -> None:
        super(Micronnet, self).__init__()

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(1, 29, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(29, 59, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(59, 74, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.fc1 = nn.Linear(1184, 300)
        self.fc2 = nn.Linear(300, nclasses)
        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv2_bn = nn.BatchNorm2d(29)
        self.conv3_bn = nn.BatchNorm2d(59)
        self.conv4_bn = nn.BatchNorm2d(74)
        self.dense1_bn = nn.BatchNorm1d(300)
        self.ReLU = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.ReLU(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.ReLU(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.ReLU(x)
        x = self.maxpool4(x)
        h = torch.flatten(x, 1)
        x = self.fc1(h)
        x = self.ReLU(x)
        x = self.dense1_bn(x)
        x = self.fc2(x)
        return h, x


class get_model(nn.Module):
    def __init__(self, weights_path: Optional[str], **model_kwargs):
        super().__init__()
        self.inner_model = Micronnet(**model_kwargs)
        self.inner_model.to(DEVICE)
        self.inner_model.to(memory_format=torch.channels_last)

        if weights_path:
            checkpoint = torch.load(weights_path, map_location=DEVICE)
            self.inner_model.load_state_dict(checkpoint)

        self.inner_model.eval()

    def forward(self, x: Tensor) -> Tensor:
        activation, output = self.inner_model(x)

        return activation, output
