import torch
from typing import Optional
import numpy as np
from art.estimators.object_tracking import PyTorchGoturn

# Load model from MITRE external repo: https://github.com/yusong-tan/pygoturn
# This needs to be defined in your config's `external_github_repo` field to be
# downloaded and placed on the PYTHONPATH
from pygoturn.src.model import GoNet  # clone from https://github.com/amoudgl/pygoturn

# load amoudgl model and instantiate ART PyTorchGoTurn model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# NOTE: PyTorchGoturn expects numpy input, not torch.Tensor input
def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchGoturn:

    model = GoNet()

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(DEVICE)

    wrapped_model = PyTorchGoturn(
        model=model,
        input_shape=(
            3,
            224,
            224,
        ),  # GoNet() uses this parameter but expects input to actually have shape (HW3)
        clip_values=(0.0, 1.0),
        channels_first=False,
        preprocessing=(
            np.array([0.485, 0.456, 0.406]),
            np.array([0.229, 0.224, 0.225]),
        ),  # ImageNet means/stds
        **wrapper_kwargs,
    )

    return wrapped_model
