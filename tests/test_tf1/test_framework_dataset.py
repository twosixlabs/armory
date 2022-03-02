"""
Test cases for framework specific ARMORY datasets.
"""

import tensorflow as tf

from armory.data import datasets
from armory import paths

DATASET_DIR = paths.DockerPaths().dataset_dir


def test_tf_generator():
    dataset = datasets.mnist(
        split="train",
        epochs=1,
        batch_size=16,
        dataset_dir=DATASET_DIR,
        framework="tf",
        preprocessing_fn=None,
        fit_preprocessing_fn=None,
    )
    assert isinstance(dataset, (tf.compat.v2.data.Dataset, tf.compat.v1.data.Dataset))
