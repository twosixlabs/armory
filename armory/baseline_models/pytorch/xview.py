"""
Pytorch Faster-RCNN for xView object detection
"""
import logging

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from art.estimators.object_detection import PyTorchFasterRCNN

from armory.data.utils import maybe_download_weights_from_s3


logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocessing_fn(img):
    img = img.transpose(0, 3, 1, 2)  # set channel_first=True

    return img


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    """
    This is an MSCOCO pre-trained model that's fine-tuned on xView.
    This model only performs inference and is not trainable. To perform other
    custom fine-tuning, please follow instructions at
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    """

    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 63)

    model.to(DEVICE)

    if weights_file:
        # TODO: get weights_file from MITRE
        filepath = maybe_download_weights_from_s3(weights_file)
        checkpoint = torch.load(filepath, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    # ART has no wrapper model for object detection
    art_model = PyTorchFasterRCNN(
        model=model,
        clip_values=(0, 255),
        channels_first=True,
        preprocessing_defences=None,  # model_kwargs?
        postprocessing_defences=None,  # model_kwargs?
        preprocessing=None,
        attack_losses=(
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),  # model_kwargs?
        device_type="cpu",  # model_kwargs?
    )

    return art_model
