"""
PyTorch Faster-RCNN Resnet50-FPN object detection model
"""
import logging
from typing import Optional

from art.estimators.object_detection import PyTorchFasterRCNN
import torch
from torchvision import models

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# NOTE: PyTorchFasterRCNN expects numpy input, not torch.Tensor input
def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchFasterRCNN:

    if weights_path:
        assert model_kwargs.get("num_classes", None) == 4, (
            "model trained on CARLA data outputs predictions for 4 classes, "
            "set model_kwargs['num_classes'] to 4."
        )
        assert not model_kwargs.get("pretrained", False), (
            "model trained on CARLA data should not use COCO-pretrained weights, set "
            "model_kwargs['pretrained'] to False."
        )

    model = models.detection.fasterrcnn_resnet50_fpn(**model_kwargs)
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchFasterRCNN(
        model, clip_values=(0.0, 1.0), channels_first=False, **wrapper_kwargs,
    )
    return wrapped_model
