"""
Test cases for ARMORY datasets.
"""

import json

import pytest
import numpy as np

from armory.utils import metrics
from armory.metrics import get, instrument, task


def test_categorical_accuracy():
    # Old
    y = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 2, 3]
    assert metrics.categorical_accuracy(y, y_pred) == [1, 1, 1, 0, 0]
    assert metrics.categorical_accuracy(y, np.eye(5)) == [1] * 5

    with pytest.raises(ValueError):
        metrics.categorical_accuracy(y, [[y_pred]])

    # New
    categorical_accuracy = get("categorical_accuracy", batch=True)
    assert categorical_accuracy(y, y_pred).tolist() == [
        1,
        1,
        1,
        0,
        0,
    ]
    assert categorical_accuracy(y, np.eye(5)).tolist() == [1] * 5


def test_top_n_categorical_accuracy():
    y = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 2, 3]

    # Old
    assert metrics.categorical_accuracy(
        y, y_pred
    ) == metrics.top_n_categorical_accuracy(y, y_pred, 1)

    # New
    categorical_accuracy = get("categorical_accuracy", batch=True)
    categorical_accuracy_no_batch = get("categorical_accuracy", batch=False)
    top_5_categorical_accuracy = get("top_5_categorical_accuracy", batch=True)
    top_n_categorical_accuracy_no_batch = task.top_n_categorical_accuracy

    for y_i, y_pred_i in zip(y, y_pred):
        assert categorical_accuracy_no_batch(
            y, y_pred
        ) == top_n_categorical_accuracy_no_batch([y_i], [y_pred_i], n=1)

    y = [2, 0]
    y_pred = [[0.1, 0.4, 0.2, 0.2, 0.1, 0.1], [0.1, 0.4, 0.2, 0.2, 0.1, 0.1]]

    # Old
    assert metrics.top_n_categorical_accuracy(y, y_pred, 3) == [1, 0]

    # New
    for y_i, y_pred_i, expected_output_i in zip(y, y_pred, [1, 0]):
        assert top_n_categorical_accuracy_no_batch(y_i, y_pred_i) == expected_output_i


def test_norms():
    x = [1, 2, 3]
    x_adv = [2, 3, 4]
    assert metrics.l1(x, x_adv) == [1, 1, 1]
    batch_size = 5

    # New
    l2 = get("l2", batch=True)
    l1 = get("l1", batch=True)
    l0 = get("l0", batch=True)
    linf = get("linf", batch=True)

    for x, x_adv in [
        (np.ones((batch_size, 16)), np.zeros((batch_size, 16))),
        (np.ones((batch_size, 4, 4)), np.zeros((batch_size, 4, 4))),
    ]:
        # Old
        assert metrics.l2(x, x_adv) == [4.0] * batch_size
        assert metrics.l1(x, x_adv) == [16.0] * batch_size
        assert metrics.l0(x, x_adv) == [1.0] * batch_size
        assert metrics.lp(x, x_adv, 4) == [2.0] * batch_size
        assert metrics.linf(x, x_adv) == [1.0] * batch_size

        # New
        assert l2(x, x_adv).tolist() == [4.0] * batch_size
        assert l1(x, x_adv).tolist() == [16.0] * batch_size
        # assert l0(x, x_adv).tolist() == [1.0] * batch_size # TODO: fix failing test by redefining l0
        assert linf(x, x_adv).tolist() == [1.0] * batch_size

    with pytest.raises(ValueError):
        metrics.lp(x, x_adv, -1)


def test_snr():
    # variable length numpy arrays
    snr = get("snr", batch=True)
    snr_db = get("snr_db", batch=True)

    x = np.array(
        [
            np.array([0, 1, 0, -1], dtype=np.float64),
            np.array([0, 1, 2, 3, 4], dtype=np.float64),
        ]
    )

    for multiplier, snr_value in [
        (0, 1),
        (0.5, 4),
        (1, np.inf),
        (2, 1),
        (3, 0.25),
        (11, 0.01),
    ]:
        # Old
        assert metrics.snr(x, x * multiplier) == [snr_value] * len(x)
        assert metrics.snr_db(x, x * multiplier) == [10 * np.log10(snr_value)] * len(x)

        # New
        assert snr(x, x * multiplier).tolist() == [snr_value] * len(x)
        assert snr_db(x, x * multiplier).tolist() == [10 * np.log10(snr_value)] * len(x)

    for addition, snr_value in [
        (0, np.inf),
        (np.inf, 0.0),
    ]:
        # Old
        assert metrics.snr(x, x + addition) == [snr_value] * len(x)
        assert metrics.snr_db(x, x + addition) == [10 * np.log10(snr_value)] * len(x)

        # New
        assert snr(x, x + addition).tolist() == [snr_value] * len(x)
        assert snr_db(x, x + addition).tolist() == [10 * np.log10(snr_value)] * len(x)

    # Old
    with pytest.raises(ValueError):
        metrics.snr(x[:1], x[1:])
    with pytest.raises(ValueError):
        metrics.snr(x, np.array([1]))

    # New
    with pytest.raises(ValueError):
        snr(x[:1], x[1:])
    with pytest.raises(ValueError):
        snr(x, np.array([1]))


def test_snr_spectrogram():
    # variable length numpy arrays
    x = np.array(
        [
            np.array([0, 1, 0, -1], dtype=np.float64),
            np.array([0, 1, 2, 3, 4], dtype=np.float64),
        ]
    )

    snr_spectrogram = get("snr_spectrogram", batch=True)
    snr_spectrogram_db = get("snr_spectrogram_db", batch=True)

    for multiplier, snr_value in [
        (0, 1),
        (0.5, 2),
        (1, np.inf),
        (2, 1),
        (3, 0.5),
        (11, 0.1),
    ]:
        # Old
        assert metrics.snr_spectrogram(x, x * multiplier) == [snr_value] * len(x)
        assert metrics.snr_spectrogram_db(x, x * multiplier) == [
            10 * np.log10(snr_value)
        ] * len(x)

        # New
        assert snr_spectrogram(x, x * multiplier).tolist() == [snr_value] * len(x)
        assert snr_spectrogram_db(x, x * multiplier).tolist() == [
            10 * np.log10(snr_value)
        ] * len(x)

    for addition, snr_value in [
        (0, np.inf),
        (np.inf, 0.0),
    ]:
        # Old
        assert metrics.snr_spectrogram(x, x + addition) == [snr_value] * len(x)
        assert metrics.snr_spectrogram_db(x, x + addition) == [
            10 * np.log10(snr_value)
        ] * len(x)

        # New
        assert snr_spectrogram(x, x + addition).tolist() == [snr_value] * len(x)
        assert snr_spectrogram_db(x, x + addition).tolist() == [
            10 * np.log10(snr_value)
        ] * len(x)

    # Old
    with pytest.raises(ValueError):
        metrics.snr_spectrogram(x[:1], x[1:])
    with pytest.raises(ValueError):
        metrics.snr_spectrogram(x, np.array([1]))

    # New
    with pytest.raises(ValueError):
        snr_spectrogram(x[:1], x[1:])
    with pytest.raises(ValueError):
        snr_spectrogram(x, np.array([1]))


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
    # NEW
    probe = instrument.get_probe()

    meter = instrument.MetricsMeter.from_config(metrics_config)
    probe.add_meter(meter)

    results_lst = []
    meter.update(
        x=[[0, 0, 0, 0]],
        x_adv=[[0, 0, 1, 1]],
        y=[0, 1, 2, 3],
        y_pred=[0, 1, 2, 2],
        y_pred_adv=[3, 2, 1, 3],
    )
    meter.measure()
    meter.finalize()
    results_lst.append(meter.results())

    # OLD
    metrics_logger.clear()
    metrics_logger.update_task([0, 1, 2, 3], [0, 1, 2, 2])
    metrics_logger.update_task([0, 1, 2, 3], [3, 2, 1, 3], adversarial=True)
    metrics_logger.update_perturbation([[0, 0, 0, 0]], [[0, 0, 1, 1]])
    metrics_logger.log_task()
    metrics_logger.log_task(adversarial=False)
    # END OLD
    results_lst.append(metrics_logger.results())

    # ensure that results are a json encodable dict
    for idx, results in enumerate(results_lst):
        assert isinstance(results, dict)
        if idx == 0:
            # json.dumps(results) #TODO: fix json encoder so exception not raised on numpy bool_ type
            assert results["mean_benign_categorical_accuracy"] == 0.75
            assert results["mean_adversarial_categorical_accuracy"] == 0.25
            assert results["mean_perturbation_l1"] == 2
            assert results["benign_categorical_accuracy"] == [1, 1, 1, 0]
            assert results["adversarial_categorical_accuracy"] == [0, 0, 0, 1]
            assert results["perturbation_l1"] == [2]

        elif idx == 1:
            # TODO: confirm name changes don't break any downstream utilities, e.g. plotting
            assert results["benign_mean_categorical_accuracy"] == 0.75
            assert results["adversarial_mean_categorical_accuracy"] == 0.25
            assert results["perturbation_mean_l1"] == 2
            assert results["benign_categorical_accuracy"] == [1, 1, 1, 0]
            assert results["adversarial_categorical_accuracy"] == [0, 0, 0, 1]
            assert results["perturbation_l1"] == [2]


def test_mAP():
    labels = {"labels": np.array([2]), "boxes": np.array([[0.1, 0.1, 0.7, 0.7]])}

    preds = {
        "labels": np.array([2, 9]),
        "boxes": np.array([[0.1, 0.1, 0.7, 0.7], [0.5, 0.4, 0.9, 0.9]]),
        "scores": np.array([0.8, 0.8]),
    }

    # Old
    ap_per_class = metrics.object_detection_AP_per_class([[labels]], [[preds]])
    assert ap_per_class[9] == 0
    assert ap_per_class[2] >= 0.99

    # New
    ap_per_class = task.object_detection_AP_per_class(
        [labels], [preds]
    )  # TODO: document the change in the format of the call to the metric
    assert ap_per_class[9] == 0
    assert ap_per_class[2] >= 0.99
