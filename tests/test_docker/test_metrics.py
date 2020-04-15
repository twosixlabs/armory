"""
Test cases for ARMORY datasets.
"""

import json

import pytest
import numpy as np

from armory.utils import metrics


def test_categorical_accuracy():
    y = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 2, 3]
    assert metrics.categorical_accuracy(y, y_pred) == [1, 1, 1, 0, 0]
    assert metrics.categorical_accuracy(y, np.eye(5)) == [1] * 5
    with pytest.raises(ValueError):
        metrics.categorical_accuracy(y, [[y_pred]])


def test_top_n_categorical_accuracy():
    y = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 2, 3]
    assert metrics.categorical_accuracy(
        y, y_pred
    ) == metrics.top_n_categorical_accuracy(y, y_pred, 1)
    y = [2, 0]
    y_pred = [[0.1, 0.4, 0.2, 0.2, 0.1, 0.1], [0.1, 0.4, 0.2, 0.2, 0.1, 0.1]]
    assert metrics.top_n_categorical_accuracy(y, y_pred, 3) == [1, 0]


def test_norms():
    x = [1, 2, 3]
    x_adv = [2, 3, 4]
    assert metrics.l1(x, x_adv) == [1, 1, 1]
    batch_size = 5
    for x, x_adv in [
        (np.ones((batch_size, 16)), np.zeros((batch_size, 16))),
        (np.ones((batch_size, 4, 4)), np.zeros((batch_size, 4, 4))),
    ]:
        assert metrics.l2(x, x_adv) == [4.0] * batch_size
        assert metrics.l1(x, x_adv) == [16.0] * batch_size
        assert metrics.l0(x, x_adv) == [16.0] * batch_size
        assert metrics.lp(x, x_adv, 4) == [2.0] * batch_size
        assert metrics.linf(x, x_adv) == [1.0] * batch_size
    with pytest.raises(ValueError):
        metrics.lp(x, x_adv, -1)


def test_metric_list():
    metric_list = metrics.MetricList("categorical_accuracy")
    metric_list.append([1], [1])
    metric_list.append([1, 2, 3], [1, 0, 2])
    assert metric_list.mean() == 0.5
    assert metric_list.values() == [1, 1, 0, 0]


def test_metrics_logger():
    metrics_config = {
        "record_metric_per_sample": True,
        "means": True,
        "perturbation": "l1",
        "task": ["categorical_accuracy"],
    }
    metrics_logger = metrics.MetricsLogger.from_config(metrics_config)
    metrics_logger.clear()
    metrics_logger.update_task([0, 1, 2, 3], [0, 1, 2, 2])
    metrics_logger.update_task([0, 1, 2, 3], [3, 2, 1, 3], adversarial=True)
    metrics_logger.update_perturbation([[0, 0, 0, 0]], [[0, 0, 1, 1]])
    metrics_logger.log_task()
    metrics_logger.log_task(adversarial=False)
    results = metrics_logger.results()

    # ensure that results are a json encodable dict
    assert isinstance(results, dict)
    json.dumps(results)
    assert results["benign_mean_categorical_accuracy"] == 0.75
    assert results["adversarial_mean_categorical_accuracy"] == 0.25
    assert results["perturbation_mean_l1"] == 2
    assert results["benign_categorical_accuracy"] == [1, 1, 1, 0]
    assert results["adversarial_categorical_accuracy"] == [0, 0, 0, 1]
    assert results["perturbation_l1"] == [2]
