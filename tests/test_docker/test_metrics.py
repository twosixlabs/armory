"""
Test cases for ARMORY datasets.
"""

import json

import pytest
import numpy as np

from armory.utils import metrics


def test_abstains():
    y = [0, 1, 2, 3, 4]
    y_pred = np.zeros((5, 10))
    y_pred[1, 1] = 1.0
    assert metrics.abstains(y, y_pred) == [1, 0, 1, 1, 1]
    for wrong_dim in np.zeros(5), np.zeros((5, 10, 10)):
        with pytest.raises(ValueError):
            metrics.abstains(y, np.zeros(5))


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
        assert metrics.l0(x, x_adv) == [1.0] * batch_size
        assert metrics.lp(x, x_adv, 4) == [2.0] * batch_size
        assert metrics.linf(x, x_adv) == [1.0] * batch_size
    with pytest.raises(ValueError):
        metrics.lp(x, x_adv, -1)


def test_snr():
    # variable length numpy arrays
    x = np.array([np.array([0, 1, 0, -1]), np.array([0, 1, 2, 3, 4]),])

    for multiplier, snr_value in [
        (0, 1),
        (0.5, 4),
        (1, np.inf),
        (2, 1),
        (3, 0.25),
        (11, 0.01),
    ]:
        assert metrics.snr(x, x * multiplier) == [snr_value] * len(x)
        assert metrics.snr_db(x, x * multiplier) == [10 * np.log10(snr_value)] * len(x)

    for addition, snr_value in [
        (0, np.inf),
        (np.inf, 0.0),
    ]:
        assert metrics.snr(x, x + addition) == [snr_value] * len(x)
        assert metrics.snr_db(x, x + addition) == [10 * np.log10(snr_value)] * len(x)

    with pytest.raises(ValueError):
        metrics.snr(x[:1], x[1:])
    with pytest.raises(ValueError):
        metrics.snr(x, np.array([1]))


def test_snr_spectrogram():
    # variable length numpy arrays
    x = np.array([np.array([0, 1, 0, -1]), np.array([0, 1, 2, 3, 4]),])

    for multiplier, snr_value in [
        (0, 1),
        (0.5, 2),
        (1, np.inf),
        (2, 1),
        (3, 0.5),
        (11, 0.1),
    ]:
        assert metrics.snr_spectrogram(x, x * multiplier) == [snr_value] * len(x)
        assert metrics.snr_spectrogram_db(x, x * multiplier) == [
            10 * np.log10(snr_value)
        ] * len(x)

    for addition, snr_value in [
        (0, np.inf),
        (np.inf, 0.0),
    ]:
        assert metrics.snr_spectrogram(x, x + addition) == [snr_value] * len(x)
        assert metrics.snr_spectrogram_db(x, x + addition) == [
            10 * np.log10(snr_value)
        ] * len(x)

    with pytest.raises(ValueError):
        metrics.snr(x[:1], x[1:])
    with pytest.raises(ValueError):
        metrics.snr(x, np.array([1]))


def test_metric_list():
    metric_list = metrics.MetricList("categorical_accuracy")
    metric_list.add_results([1], [1])
    metric_list.add_results([1, 2, 3], [1, 0, 2])
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


def test_mAP():
    labels = {"labels": np.array([2]), "boxes": np.array([[0.1, 0.1, 0.7, 0.7]])}

    preds = {
        "labels": np.array([2, 9]),
        "boxes": np.array([[0.1, 0.1, 0.7, 0.7], [0.5, 0.4, 0.9, 0.9]]),
        "scores": np.array([0.8, 0.8]),
    }

    ap_per_class = metrics.object_detection_AP_per_class([labels], [preds])
    assert ap_per_class[9] == 0
    assert ap_per_class[2] >= 0.99


def test_object_detection_metrics():
    y = [
        {
            "labels": np.array([2, 7, 6]),
            "boxes": np.array(
                [[0.1, 0.1, 0.7, 0.7], [0.3, 0.3, 0.4, 0.4], [0.05, 0.05, 0.15, 0.15]]
            ),
        }
    ]

    y_pred = [
        {
            "labels": np.array([2, 9, 3]),
            "boxes": np.array(
                [
                    [0.12, 0.09, 0.68, 0.7],
                    [0.5, 0.4, 0.9, 0.9],
                    [0.05, 0.05, 0.15, 0.15],
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8]),
        }
    ]
    score_threshold = 0.5
    iou_threshold = 0.5
    (
        true_positive_rate_per_img,
        misclassification_rate_per_img,
        disappearance_rate_per_img,
        hallucinations_per_img,
    ) = metrics._object_detection_get_tpr_mr_dr_hr(
        y, y_pred, score_threshold=score_threshold, iou_threshold=iou_threshold
    )
    for rate_per_img in [
        true_positive_rate_per_img,
        misclassification_rate_per_img,
        disappearance_rate_per_img,
    ]:
        assert isinstance(rate_per_img, list)
        assert len(rate_per_img) == 1
        assert np.abs(rate_per_img[0] - 1.0 / 3.0) < 0.001  # all 3 rates should be 1/3
    assert isinstance(hallucinations_per_img, list)
    assert len(hallucinations_per_img) == 1
    assert hallucinations_per_img[0] == 1
