"""
ResNet18 CNN model for NxNx3 image classification
"""
from typing import Optional

from art.estimators.classification import PyTorchClassifier
import torch
from torchvision import models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OuterModel(torch.nn.Module):
    def __init__(
        self,
        weights_path: Optional[str],
        **model_kwargs,
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

    lr = wrapper_kwargs.pop("learning_rate", 0.1)

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        input_shape=wrapper_kwargs.pop(
            "input_shape", (224, 224, 3)
        ),  # default to imagenet shape
        channels_first=False,
        **wrapper_kwargs,
        clip_values=(0.0, 1.0),
    )
    return wrapped_model
