"""
Test cases for Armory metrics.
"""

import math

import pytest
import numpy as np

from armory.utils import metrics

# Mark all tests in this file as `unit`
pytestmark = pytest.mark.unit


@pytest.mark.docker_required
@pytest.mark.pytorch_deepspeech
@pytest.mark.slow
def test_entailment():
    """
    Slow due to 1 GB file download and multiple model predictions
    """
    metric = metrics.Entailment()
    metric_repeat = metrics.Entailment()
    assert metric.model is metric_repeat.model

    from armory.attacks.librispeech_target_labels import (
        ground_truth_100,
        entailment_100,
    )

    num_samples = 100
    assert len(ground_truth_100) == num_samples
    assert len(entailment_100) == num_samples

    from collections import Counter

    label_mapping = ["contradiction", "neutral", "entailment"]
    gt_gt = metric(ground_truth_100, ground_truth_100)
    gt_gt = [label_mapping[i] if i in (0, 1, 2) else i for i in gt_gt]
    c = Counter()
    c.update(gt_gt)
    assert c["entailment"] == num_samples

    gt_en = metric(ground_truth_100, entailment_100)
    gt_en = [label_mapping[i] if i in (0, 1, 2) else i for i in gt_en]
    c = Counter()
    c.update(gt_en)
    # NOTE: currently, i=6 is entailment and i=38 is neutral
    # TODO: update entailment_100 to make both entailment, then update >= 98 to == 100
    #     >>> ground_truth_100[6]
    #     'THIS WAS WHAT DID THE MISCHIEF SO FAR AS THE RUNNING AWAY WAS CONCERNED'
    #     >>> entailment_100[6]
    #     'THIS WAS WHAT DID THE MISCHIEF SO FAR AS THE WALKING AWAY WAS CONCERNED'
    #     >>> ground_truth_100[38]
    #     'MAY WE SEE GATES AT ONCE ASKED KENNETH
    #     >>> entailment_100[38]
    #     'MAY WE SEE GATES TOMORROW ASKED KENNETH'
    assert c["contradiction"] >= 98


def test_total_entailment(caplog):
    for invalid_results in (["invalid name"], [-1], [3], [None]):
        with pytest.raises(ValueError):
            metrics.total_entailment(invalid_results)

    metrics.total_entailment([0, 1, 2])
    assert "Entailment outputs are (0, 1, 2) ints, not strings, mapping" in caplog.text

    results = metrics.total_entailment(["contradiction"])
    assert results == dict(contradiction=1, neutral=0, entailment=0)
    assert isinstance(results, dict)

    results = metrics.total_entailment([0, 1, 1, "neutral", "entailment", "entailment"])
    assert results == dict(contradiction=1, neutral=3, entailment=2)
    assert sum(results.values()) == 6


def test_tpr_fpr():
    actual_conditions = [0] * 10 + [1] * 10
    predicted_conditions = [0] * 4 + [1] * 6 + [0] * 3 + [1] * 7
    results = metrics.tpr_fpr(actual_conditions, predicted_conditions)
    for k, v in [
        ("true_positives", 7),
        ("true_negatives", 4),
        ("false_positives", 6),
        ("false_negatives", 3),
        ("true_positive_rate", 0.7),
        ("true_negative_rate", 0.4),
        ("false_positive_rate", 0.6),
        ("false_negative_rate", 0.3),
        ("f1_score", 7 / (7 + 0.5 * (6 + 3))),
    ]:
        assert results[k] == v, "k"

    results = metrics.tpr_fpr([], [])
    for k, v in [
        ("true_positives", 0),
        ("true_negatives", 0),
        ("false_positives", 0),
        ("false_negatives", 0),
        ("true_positive_rate", float("nan")),
        ("true_negative_rate", float("nan")),
        ("false_positive_rate", float("nan")),
        ("false_negative_rate", float("nan")),
        ("f1_score", float("nan")),
    ]:
        if math.isnan(v):
            assert math.isnan(results[k]), f"{k}"
        else:
            assert results[k] == v, f"{k}"

    with pytest.raises(ValueError):
        metrics.tpr_fpr(actual_conditions, [predicted_conditions])
    for array in 0, [[0, 1], [1, 0]]:
        with pytest.raises(ValueError):
            metrics.tpr_fpr(array, array)


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


def test_video_tracking_metrics():
    # Recall that first box isn't counted towards metrics. Second box has IoU of 1.0,
    # third box has IoU of 0
    y = [
        {
            "boxes": np.array(
                [[0.0, 0.0, 0.5, 0.5], [0.1, 0.0, 0.6, 0.5], [0.0, 0.0, 0.3, 0.3]]
            )
        }
    ]
    y_pred = [
        {
            "boxes": np.array(
                [[0.0, 0.0, 0.5, 0.5], [0.1, 0.0, 0.6, 0.5], [0.8, 0.8, 1.0, 1.0]]
            )
        }
    ]

    mean_iou = metrics.video_tracking_mean_iou(y, y_pred)
    mean_success = metrics.video_tracking_mean_success_rate(y, y_pred)

    for result in [mean_success, mean_iou]:
        assert isinstance(result, list)
        assert len(result) == len(y)
        assert result[0] == 0.5
