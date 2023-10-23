# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module with implementation of PyTorch networks."""

import math
import os

import torch


google_root = "/Users/david.slater/google"

path_map = {
    "baseline": "checkpoints/baseline/final_checkpoint-1",
}


def load(name="baseline", root=google_root):
    checkpoint = path_map.get(name)
    if checkpoint is None:
        raise ValueError(f"{name} has no checkpoint")

    convnet = AllConvModelTorch(
        num_classes=10,
        num_filters=64,
        input_shape=[3, 32, 32],
    )

    full_path = os.path.join(root, checkpoint + ".torchmodel")
    convnet.load_state_dict(torch.load(full_path))
    return convnet


class GlobalAveragePool(torch.nn.Module):
    """Global average pooling operation."""

    def forward(self, x):
        return torch.mean(x, axis=[2, 3])


class AllConvModelTorch(torch.nn.Module):
    """All convolutional network architecture."""

    def __init__(
        self, num_classes, num_filters, input_shape, activation=torch.nn.LeakyReLU(0.2)
    ):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=(1, 1))

        self.layers = torch.nn.ModuleList([])
        prev = input_shape[0]
        log_resolution = int(round(math.log(input_shape[1]) / math.log(2)))
        for scale in range(log_resolution - 2):
            self.layers.append(torch.nn.Conv2d(prev, num_filters << scale, **conv_args))
            self.layers.append(activation)
            prev = num_filters << (scale + 1)
            self.layers.append(torch.nn.Conv2d(num_filters << scale, prev, **conv_args))
            self.layers.append(activation)
            self.layers.append(torch.nn.AvgPool2d((2, 2)))
        self.layers.append(
            torch.nn.Conv2d(prev, num_classes, kernel_size=3, padding=(1, 1))
        )
        self.layers.append(GlobalAveragePool())
        self.layers.append(torch.nn.Softmax(dim=1))

    def __call__(self, x, training=False):
        del training  # ignore training argument since don't have batch norm
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        for layer in self.layers:
            x = layer(x)
        return x
