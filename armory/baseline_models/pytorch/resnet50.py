"""
ResNet50 CNN model for 244x244x3 image classification
"""
import logging
from typing import Optional

from art.classifiers import PyTorchClassifier
import numpy as np
import torch
from torchvision import models


logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDEV = [0.229, 0.224, 0.225]


def preprocessing_fn(img: np.ndarray):
    """
    Standardize, then normalize imagenet images
    """
    # Standardize images to [0, 1]
    img /= 255.0

    # Normalize images ImageNet means
    for i, (mean, std) in enumerate(zip(IMAGENET_MEANS, IMAGENET_STDEV)):
        img[i] -= mean
        img[i] /= std

    return img


# NOTE: PyTorchClassifier expects numpy input, not torch.Tensor input
def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = models.resnet50(**model_kwargs)
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(224, 224, 3),
        **wrapper_kwargs,
        clip_values=(
            np.array(
                [
                    0.0 - IMAGENET_MEANS[0] / IMAGENET_STDEV[0],
                    0.0 - IMAGENET_MEANS[1] / IMAGENET_STDEV[1],
                    0.0 - IMAGENET_MEANS[2] / IMAGENET_STDEV[2],
                ]
            ),
            np.array(
                [
                    1.0 - IMAGENET_MEANS[0] / IMAGENET_STDEV[0],
                    1.0 - IMAGENET_MEANS[1] / IMAGENET_STDEV[1],
                    1.0 - IMAGENET_MEANS[2] / IMAGENET_STDEV[2],
                ]
            ),
        ),
    )
    return wrapped_model
