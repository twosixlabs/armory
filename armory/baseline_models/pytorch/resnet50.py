import logging
import os

from art.classifiers import PyTorchClassifier
import numpy as np
import torch
from torchvision import models

from armory import paths
from armory.data.utils import download_file_from_s3


logger = logging.getLogger(__name__)
os.environ["TORCH_HOME"] = os.path.join(paths.docker().dataset_dir, "pytorch", "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDEV = [0.229, 0.224, 0.225]


def preprocessing_fn(img):
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
def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = models.resnet50(**model_kwargs)
    model.to(DEVICE)

    if weights_file:
        saved_model_dir = paths.docker().saved_model_dir
        filepath = os.path.join(saved_model_dir, weights_file)

        if not os.path.isfile(filepath):
            download_file_from_s3(
                "armory-public-data",
                f"model-weights/{weights_file}",
                f"{saved_model_dir}/{weights_file}",
            )

        checkpoint = torch.load(filepath, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

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
