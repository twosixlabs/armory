"""
CNN model for 28x28x1 image classification
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from art.classifiers import PyTorchClassifier

from armory.data.utils import maybe_download_weights_from_s3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocessing_fn(img):
    # Model will trained with inputs normalized from 0 to 1
    img = img.astype(np.float32) / 255.0
    return img


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, 1)
        self.conv2 = nn.Conv2d(4, 10, 5, 1)
        self.fc1 = nn.Linear(160, 100)
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


def make_mnist_model(**kwargs):
    return Net()


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = make_mnist_model(**model_kwargs)
    model.to(DEVICE)

    if weights_file:
        filepath = maybe_download_weights_from_s3(weights_file)
        checkpoint = torch.load(filepath, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
