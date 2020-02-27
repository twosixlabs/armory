import logging
import os

from art.classifiers import PyTorchClassifier
import numpy as np
import torch
from torch import nn
from torchvision import models

from armory import paths


logger = logging.getLogger(__name__)
os.environ["TORCH_HOME"] = os.path.join(paths.docker().dataset_dir, "pytorch", "models")


def resnet50(pretrained=True):
    return models.resnet50(pretrained=pretrained)


def preprocessing_fn(X, inplace=False):
    """
    Standardize, then normalize imagenet images

    Reimplemented from torchvision.transforms.Normalize to:
        a) work with both torch tensors and numpy arrays
        b) enable 4D array Batch-RGB inputs consistent with our data loader
    """
    is_numpy, is_torch = False, False
    if isinstance(X, np.ndarray):
        is_numpy = True
    if isinstance(X, torch.Tensor):
        is_torch = True
    if is_numpy and is_torch:
        raise ValueError("multiple inheritance of numpy and torch not supported")
    if not is_numpy and not is_torch:
        raise ValueError(
            f"type(X) is {type(X)}. Only np.ndarray and torch.Tensor supported"
        )

    inplace = bool(inplace)
    if X.min() < 0 or X.max() <= 1:
        logger.warning(
            "Input is not in expected range [0, 255], and may already be normalized"
        )
    if X.ndim != 4:
        raise ValueError(f"X.ndim must be 4, not {X.ndim}")

    if X.shape[1:] == (224, 224, 3):
        # Height-Width-Channel to Channel-Height-Width
        if inplace:
            raise ValueError("Cannot reorder tensor and modify inplace")
        ordering = (0, 3, 1, 2)
        if is_numpy:
            X = np.transpose(X, axes=ordering)
        else:  # is_torch
            X = X.permute(*ordering)

    elif X.shape[1:] != (3, 224, 224):
        raise ValueError(f"Input shape {X.shape[1:]} is invalid")

    if not inplace:
        if is_numpy:
            X = X.copy()
        else:  # is_torch
            X = X.clone()

    # Standardize images to [0, 1]
    X /= 255.0

    # Normalize images ImageNet means
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for i, (mean, std) in enumerate(zip(means, stds)):
        X[i] -= mean
        X[i] /= std

    return X


# NOTE: PyTorchClassifier expects numpy input, not torch.Tensor input
def get_art_model(model_kwargs, wrapper_kwargs):
    model = resnet50(**model_kwargs)
    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=(224, 224, 3),
        **wrapper_kwargs,
    )
    return wrapped_model
