"""
PyTorch Faster-RCNN Resnet50-FPN object detection model
"""
from collections import OrderedDict
from typing import Optional

from art.estimators.object_detection import PyTorchFasterRCNN
import torch
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MOEFusion(torch.nn.Module):
    """
    Mixture-of-expert fusion that follows the paper by Yang et al. (CVPR 2021)
    Paper link: https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Defending_Multimodal_Fusion_Models_Against_Single-Source_Adversaries_CVPR_2021_paper.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        modalities,
        hidden_channels=512,
        feat_map_names=["0", "1", "2", "3", "pool"],
    ):
        super(MOEFusion, self).__init__()

        self.fus_in_channels = in_channels
        self.fus_hidden_channels = hidden_channels
        self.fus_out_channels = out_channels
        self.modalities = modalities

        self.gating_network = torch.nn.ModuleDict(
            {name: self.construct_gating_network() for name in feat_map_names}
        )
        self.expert_classifiers = self.construct_expert_classifiers()

    def construct_gating_network(self):
        """From Section "3.2. Robust Feature Fusion Layer" in DMF paper

        This layer consists of an ensemble of k + 1 feature fusion operations e_{1}, ..., e_{k+1},
        each of which is specialized to exclude one modality, as illustrated in Figure 2(c).
        Formally, each fusion operation takes the multimodal features z as input and
        performs a fusion of a subset of the features...
        """
        net = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.fus_in_channels * len(self.modalities),
                self.fus_hidden_channels,
                kernel_size=1,
                stride=1,
            ),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(
                self.fus_hidden_channels,
                len(self.modalities) + 1,
                kernel_size=1,
                stride=1,
            ),
        )
        return net

    def construct_expert_classifiers(self):
        expert_classifiers = {}
        """
        Use pass-through features for 'rgb' or 'depth' single modalities,
        i.e., only create an expert classifier for multiple fused modalities
        """
        expert_classifiers["rgb_depth"] = torch.nn.Conv2d(
            2 * self.fus_in_channels, self.fus_out_channels, kernel_size=1, stride=1
        )
        return torch.nn.ModuleDict(expert_classifiers)

    def apply_gating_network(self, features, feat_map_name):
        # features is a dict with k=mod, v=Tensor (feature map)
        concat_features = torch.cat([features[mod] for mod in self.modalities], 1)
        return self.gating_network[feat_map_name](concat_features)

    def forward(self, features, feat_map_name=None):
        """
        features is a dict with k=mod, v=Tensor (feature map)
        """
        expert_gate_logits = self.apply_gating_network(features, feat_map_name)
        expert_gates = F.softmax(expert_gate_logits, 1).unsqueeze(
            1
        )  # values between 0 and 1

        outputs = []
        for mod_subset in ["rgb_depth", "rgb", "depth"]:
            subset_feats = torch.cat(
                [features[mod] for mod in mod_subset.split("_")], dim=1
            )
            if mod_subset == "rgb_depth":
                subset_feats = self.expert_classifiers[mod_subset](subset_feats)
            outputs.append(subset_feats)

        outputs = torch.stack(outputs, dim=2)
        outputs = torch.sum(outputs * expert_gates, dim=2)  # equation (5) in DMF paper

        return outputs, expert_gate_logits


class MultimodalRobust(torch.nn.Module):
    """
    Multimodal model with robust fusion that follows the paper by Yang et al. (CVPR 2021)
    Paper link: https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Defending_Multimodal_Fusion_Models_Against_Single-Source_Adversaries_CVPR_2021_paper.pdf
    """

    def __init__(self, modalities=["rgb", "depth"]):
        super(MultimodalRobust, self).__init__()

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
        fus_in_channels = self.out_channels
        fusion_net = MOEFusion(fus_in_channels, self.out_channels, self.modalities)
        return fusion_net

    def forward(self, stacked_input):
        # "un-stack/un-concat" the modalities
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

            fused_features, _ = self.fusion_net(feature_layer, feat_map_name=layer_name)
            output.update({layer_name: fused_features})

        return output  # <- OrderedDict[str, torch.Tensor]


# NOTE: PyTorchFasterRCNN expects numpy input, not torch.Tensor input
def get_art_model_mm_robust(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchFasterRCNN:

    num_classes = model_kwargs.pop("num_classes", 3)

    backbone = MultimodalRobust(**model_kwargs)

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
