"""
ResNet50 CNN model for 244x244x3 image classification
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
        super().__init__()
        self.inner_model = models.resnet50(**model_kwargs)
        self.inner_model.to(DEVICE)

        if weights_path:
            checkpoint = torch.load(weights_path, map_location=DEVICE)
            self.inner_model.load_state_dict(checkpoint)

        self.imagenet_means = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32, device=DEVICE
        )
        self.imagenet_stdev = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=DEVICE
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = ((x - self.imagenet_means) / self.imagenet_stdev).permute(0, 3, 1, 2)
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
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(224, 224, 3),
        channels_first=False,
        **wrapper_kwargs,
        clip_values=(0.0, 1.0),
    )
    return wrapped_model
