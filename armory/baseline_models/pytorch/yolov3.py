from typing import Optional

from art.estimators.object_detection import PyTorchYolo
from pytorchyolo.models import load_model
from pytorchyolo.utils.loss import compute_loss
import torch

from armory.baseline_models import model_configs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Yolo(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, targets=None):
        if self.training:
            outputs = self.model(x)
            loss, _ = compute_loss(outputs, targets, self.model)
            loss_components_dict = {"loss_total": loss}
            return loss_components_dict
        else:
            out = self.model(x)
            return out


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchYolo:

    model_kwargs["model_path"] = model_configs.get_path(model_kwargs["model_path"])
    model = load_model(weights_path=weights_path, **model_kwargs)
    model_wrapper = Yolo(model)

    params = [p for p in model_wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True)

    detector = PyTorchYolo(
        model=model_wrapper,
        device_type=DEVICE,
        input_shape=(416, 416, 3),
        optimizer=optimizer,
        clip_values=(0, 1),
        channels_first=False,
        attack_losses=("loss_total",),
        **wrapper_kwargs,
    )
    return detector
