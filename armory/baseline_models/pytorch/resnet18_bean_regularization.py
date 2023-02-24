"""
ResNet18 model to be used for data interpretations
"""
from typing import Optional

import torch
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import ResNet as resnet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet(resnet):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__(
            block=block, layers=layers, num_classes=num_classes
        )

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        h = torch.flatten(x, 1)
        x = self.fc(h)
        return h, x


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


class get_model(torch.nn.Module):
    def __init__(
        self,
        weights_path: Optional[str],
        **model_kwargs,
    ):
        # default to imagenet mean and std
        data_means = model_kwargs.pop("data_means", [0.485, 0.456, 0.406])
        data_stds = model_kwargs.pop("data_stds", [0.229, 0.224, 0.225])

        super().__init__()
        self.inner_model = ResNet18(**model_kwargs)
        self.inner_model.to(DEVICE)

        if weights_path:
            checkpoint = torch.load(weights_path, map_location=DEVICE)
            self.inner_model.load_state_dict(checkpoint)

        self.inner_model.eval()

        self.data_means = torch.tensor(data_means, dtype=torch.float32, device=DEVICE)
        self.data_stdev = torch.tensor(data_stds, dtype=torch.float32, device=DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = ((x - self.data_means) / self.data_stdev).permute(0, 3, 1, 2)
        activation, output = self.inner_model(x_norm)

        return activation, output
