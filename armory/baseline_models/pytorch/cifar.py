import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from art.classifiers import PyTorchClassifier

from armory import paths
from armory.data.utils import download_file_from_s3


def preprocessing_fn(img):
    # Model will trained with inputs normalized from 0 to 1
    img = img.astype(np.float32) / 255.0
    return img


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 5, 1)
        self.conv2 = nn.Conv2d(4, 10, 5, 1)
        self.fc1 = nn.Linear(250, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # from NHWC to NCHW
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def make_cifar_model(**kwargs):
    return Net()


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = make_cifar_model(**model_kwargs)

    if torch.cuda.is_available():
        model.cuda()

    if weights_file:
        saved_model_dir = paths.docker().saved_model_dir
        filepath = os.path.join(saved_model_dir, weights_file)

        if not os.path.isfile(filepath):
            download_file_from_s3(
                "armory-public-data",
                f"model-weights/{weights_file}",
                f"{saved_model_dir}/{weights_file}",
            )

        model.load(filepath)

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
