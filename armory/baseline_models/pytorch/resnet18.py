"""
ResNet18 CNN model for NxNx3 image classification
"""
import logging
from typing import Optional

from art.classifiers import PyTorchClassifier
import torch
from torchvision import models


logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OuterModel(torch.nn.Module):
    def __init__(
        self, weights_path: Optional[str], **model_kwargs,
    ):
        # default to imagenet mean and std
        data_means = model_kwargs.pop("data_means", [0.485, 0.456, 0.406])
        data_stds = model_kwargs.pop("data_stds", [0.229, 0.224, 0.225])

        super().__init__()
        self.inner_model = models.resnet18(**model_kwargs)
        self.inner_model.to(DEVICE)

        if weights_path:
            checkpoint = torch.load(weights_path, map_location=DEVICE)
            self.inner_model.load_state_dict(checkpoint)

        self.data_means = torch.tensor(data_means, dtype=torch.float32, device=DEVICE)
        self.data_stdev = torch.tensor(data_stds, dtype=torch.float32, device=DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = ((x - self.data_means) / self.data_stdev).permute(0, 3, 1, 2)
        output = self.inner_model(x_norm)

        return output


# NOTE: PyTorchClassifier expects numpy input, not torch.Tensor input
def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:

    model = OuterModel(weights_path=weights_path, **model_kwargs)

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9),
        input_shape=wrapper_kwargs.pop(
            "input_shape", (224, 224, 3)
        ),  # default to imagenet shape
        channels_first=False,
        **wrapper_kwargs,
        clip_values=(0.0, 1.0),
    )
    return wrapped_model
