"""
Test cases for poisoning metrics
"""

import copy
from importlib import import_module

import numpy as np
import pytest

from armory.data.utils import maybe_download_weights_from_s3
from armory.metrics import poisoning

pytestmark = pytest.mark.unit


@pytest.mark.slow
def test_explanatory_model():

    config_keys = [
        "cifar10_silhouette_model",
        "gtsrb_silhouette_model",
        "resisc10_silhouette_model",
    ]
    data_sizes = [
        (10, 32, 32, 3),
        (10, 48, 48, 3),
        (10, 256, 256, 3),
    ]
    activation_shapes = [
        (10, 512),
        (10, 1184),
        (10, 512),
    ]

    for config_key, data_size, activation_shape in zip(
        config_keys, data_sizes, activation_shapes
    ):

        config = poisoning.EXPLANATORY_MODEL_CONFIGS[config_key]
        config = copy.copy(config)

        # Test once with from_config()
        model = poisoning.ExplanatoryModel.from_config(config)
        x = np.random.rand(*data_size).astype(np.float32)
        activations = model.get_activations(x)

        assert activations.shape == activation_shape

        # Then test with class constructor
        module = config.pop("module")
        name = config.pop("name")
        weights_file = config.pop("weights_file")
        model_kwargs = config.pop("model_kwargs", {})

        weights_path = maybe_download_weights_from_s3(
            weights_file, auto_expand_tars=True
        )
        model_module = import_module(module)
        model_fn = getattr(model_module, name)
        model_ = model_fn(weights_path, **model_kwargs)
        model = poisoning.ExplanatoryModel(model_, **config)
        activations = model.get_activations(x)

        assert activations.shape == activation_shape


def test_preprocess():

    x = np.random.rand(10, 32, 32, 3).astype(np.float32)
    x_ = poisoning.ExplanatoryModel._preprocess(x)
    assert x_.shape == (10, 224, 224, 3)
    assert x_.max() <= 1
    assert x_.min() >= 0
