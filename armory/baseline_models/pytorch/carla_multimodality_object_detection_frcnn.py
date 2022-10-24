"""
PyTorch Faster-RCNN Resnet50-FPN object detection model
"""
from collections import OrderedDict
from typing import Optional

from art.estimators.object_detection import PyTorchFasterRCNN
import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConcatFusion(torch.nn.Module):
    """
    For integration, we're just going to pass through the RGB features.
    Will update with the final code soon.
    """

    def __init__(self, in_channels, out_channels, modalities):
        super(ConcatFusion, self).__init__()

        self.fus_in_channels = in_channels
        self.fus_out_channels = out_channels
        self.modalities = modalities
        self.net = torch.nn.Conv2d(in_channels, out_channels, 1, stride=1)

    def forward(self, features):
        # concat features on the channel dimension; tensor is [b_size, C, H, W]
        concat_features = torch.cat([features[mod] for mod in self.modalities], 1)
        out = self.net(concat_features)
        return out


class MultimodalNaive(torch.nn.Module):
    def __init__(self, modalities=["rgb", "depth"]):
        super(MultimodalNaive, self).__init__()

        self.modalities = modalities
        self.out_channels = 256

        # type(feature_nets) == nn.ModuleDict; k=mod,v=nn.Module
        self.feature_nets = self.construct_feature_nets()
        self.fusion_net = self.construct_fusion_net()

    def construct_feature_nets(self):
        feature_nets = {}
        for mod in self.modalities:
            mod_net = resnet_fpn_backbone("resnet50", pretrained=False)

            feature_nets[mod] = mod_net

        feature_nets = torch.nn.ModuleDict(feature_nets)
        return feature_nets

    def construct_fusion_net(self):
        # in_channels = 1024 * len(self.modalities) <- DMF paper
        # and then in ConcatFusion, nn.Conv2d(in_channels=in_channels)
        in_channels = self.out_channels * len(self.modalities)
        fusion_net = ConcatFusion(in_channels, self.out_channels, self.modalities)
        return fusion_net

    def forward(self, stacked_input):  # -> OrderedDict[str, torch.Tensor]:
        # "un-stack/un-concat" the modalities, assuming RGB are in first 3 channels
        inputs = {}
        for i, mod in enumerate(self.modalities, 1):
            l_bound = (i - 1) * 3  # indicates where to "start slicing" (inclusive)
            u_bound = i * 3  # indicates where to "stop slicing" (exclusive)
            inputs[mod] = stacked_input[:, l_bound:u_bound, :, :]

        # extract features for each input
        features = {}
        for mod in self.modalities:
            # features[mod] will be an OrderedDict of (5) feature maps (1 for each layer)
            features[mod] = self.feature_nets[mod](inputs[mod])

        # build list containing dict for each layer returned from resnet_fpn_backbone
        feature_layers = [dict() for i in range(len(features[mod]))]
        for mod in self.modalities:
            for idx, (k, v) in enumerate(features[mod].items()):
                feature_layers[idx].update({mod: (k, v)})

        # pass each respective layer through fusion
        # k==modalities, v==respective ordered dict of feature maps!

        output = OrderedDict()
        # pass through fusion
        for layer in feature_layers:
            # layer is a dict with k=mod, v=tuple(layer_name, layer_feature_map)
            layer_name = layer[self.modalities[0]][
                0
            ]  # set name in OrderedDict; this is "hacky"/non-intuitive
            feature_layer = {
                k: v[1] for k, v in layer.items()
            }  # dict with k=mod, v=Tensor (feature map)
            fused_features = self.fusion_net(
                feature_layer
            )  # fusion layer should output a Tensor
            output.update({layer_name: fused_features})

        return output


# NOTE: PyTorchFasterRCNN expects numpy input, not torch.Tensor input
def get_art_model_mm(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchFasterRCNN:

    num_classes = model_kwargs.pop("num_classes", 3)

    backbone = MultimodalNaive(**model_kwargs)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        image_mean=[0.485, 0.456, 0.406, 0.0, 0.0, 0.0],
        image_std=[0.229, 0.224, 0.225, 1.0, 1.0, 1.0],
    )
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchFasterRCNN(
        model,
        clip_values=(0.0, 1.0),
        channels_first=False,
        **wrapper_kwargs,
    )
    return wrapped_model
