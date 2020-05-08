"""
Test cases for framework specific ARMORY datasets.
"""

import pytest

from armory.data import datasets
from armory import paths

DATASET_DIR = paths.DockerPaths().dataset_dir


def test_pytorch_generator():
    with pytest.raises(NotImplementedError):
        _ = datasets.mnist(
            split_type="train",
            epochs=1,
            batch_size=16,
            dataset_dir=DATASET_DIR,
            framework="pytorch",
        )
