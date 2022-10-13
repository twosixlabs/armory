"""
PyTorch Faster-RCNN Resnet50-FPN object detection model
"""
from typing import Optional

from art.estimators.object_detection import PyTorchFasterRCNN
import torch
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# NOTE: PyTorchFasterRCNN expects numpy input, not torch.Tensor input
def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchFasterRCNN:

    if weights_path:
        assert not model_kwargs.get("pretrained", False), (
            "model trained on CARLA data should not use COCO-pretrained weights, set "
            "model_kwargs['pretrained'] to False."
        )

    model = models.detection.fasterrcnn_resnet50_fpn(**model_kwargs)
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        assert (
            "roi_heads.box_predictor.cls_score.bias" in checkpoint
        ), "invalid checkpoint for current model, layers do no match."
        assert (
            model.roi_heads.box_predictor.cls_score.out_features
            == checkpoint["roi_heads.box_predictor.cls_score.bias"].shape[0]
        ), (
            f"provided model checkpoint does not match supplied model_kwargs['num_classes']: "
            f"{model_kwargs['num_classes']} != {checkpoint['roi_heads.box_predictor.cls_score.bias'].shape[0]}"
        )
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchFasterRCNN(
        model,
        clip_values=(0.0, 1.0),
        channels_first=False,
        **wrapper_kwargs,
    )
    return wrapped_model
