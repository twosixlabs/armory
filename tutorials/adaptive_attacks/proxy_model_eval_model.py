from typing import Optional

from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn

from armory.baseline_models.pytorch.cifar import Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModifiedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Net()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.net.forward(x)
        else:
            x_mod = torch.round(x * 256.0) / 256.0
            return self.net.forward(x_mod)


def make_modified_model(**kwargs) -> ModifiedNet:
    return ModifiedNet()


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = make_modified_model(**model_kwargs)
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(32, 32, 3),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
