"""
Test cases for poisoning metrics
"""

from importlib import import_module
import copy
import pytest

import numpy as np

from armory.metrics import poisoning
from armory.data.utils import maybe_download_weights_from_s3

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


def test_model_and_filter_bias():

    fairness = poisoning.FairnessMetrics(
        {"explanatory_model": "cifar10_silhouette_model"}
    )

    np.random.seed(1)
    x_poison = np.random.rand(11, 32, 32, 3).astype(np.float32)
    y_poison = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    poison_index = np.array([8, 9])
    predicted_clean_indices = np.array([1, 2, 4, 5, 7, 8])
    test_x = np.random.rand(15, 32, 32, 3).astype(np.float32)
    test_y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    test_y_pred = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )
    # test_y_pred = np.array([0,1,0,2,2,2,0,1,2,2,2])
    train_set_class_labels = np.array([0, 1, 2])
    test_set_class_labels = np.array([0, 1, 2])
    is_filtering_defense = True

    poisoned_mask = fairness.get_poisoned_mask(y_poison, poison_index)
    (
        majority_mask_train_unpoisoned,
        majority_ceilings,
    ) = fairness.get_train_majority_mask_and_ceilings(x_poison, y_poison, poisoned_mask)
    majority_mask_test_set = fairness.get_test_majority_mask(
        test_x, test_y, majority_ceilings
    )

    model_bias_answers = {
        "model_bias_chi^2_p_value_00": 0.3613104285261789,
        "model_bias_spd_00": 0.5,
        "model_bias_chi^2_p_value_01": 0.17090352023079358,
        "model_bias_spd_01": 0.75,
        "model_bias_chi^2_p_value_02": 0.3613104285261789,
        "model_bias_spd_02": -0.5,
    }
    filter_bias_answers = {
        "filter_bias_chi^2_p_value_00": 0.3864762307712325,
        "filter_bias_spd_00": -0.5,
        "filter_bias_chi^2_p_value_01": 0.0832645166635504,
        "filter_bias_spd_01": 1.0,
        "filter_bias_chi^2_p_value_02": 0.3864762307712325,
        "filter_bias_spd_02": -0.5,
    }

    fairness.compute_model_bias(
        test_y, test_y_pred, majority_mask_test_set, test_set_class_labels
    )
    for key in model_bias_answers:
        assert fairness.results[key] == pytest.approx(model_bias_answers[key])

    fairness.compute_filter_bias(
        y_poison,
        predicted_clean_indices,
        poisoned_mask,
        majority_mask_train_unpoisoned,
        train_set_class_labels,
    )
    for key in filter_bias_answers:
        assert fairness.results[key] == pytest.approx(filter_bias_answers[key])
