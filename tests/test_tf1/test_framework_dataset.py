"""
Test cases for framework specific ARMORY datasets.
"""

import tensorflow as tf

from armory.data import datasets
from armory import paths

DATASET_DIR = paths.DockerPaths().dataset_dir


def test_tf_generator():
    dataset = datasets.mnist(
        split_type="train",
        epochs=1,
        batch_size=16,
        dataset_dir=DATASET_DIR,
        framework="tf",
    )
    assert isinstance(dataset, tf.data.Dataset)
