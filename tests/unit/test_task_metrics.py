"""
Test cases for Armory metrics.
"""

import math

import pytest
import numpy as np

from armory.metrics import task

pytestmark = pytest.mark.unit


@pytest.mark.docker_required
@pytest.mark.pytorch_deepspeech
@pytest.mark.slow
def test_entailment():
    """
    Slow due to 1 GB file download and multiple model predictions
    """
    metric = task.Entailment()
    metric_repeat = task.Entailment()
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
            task.total_entailment(invalid_results)

    task.total_entailment([0, 1, 2])
    assert "Entailment outputs are (0, 1, 2) ints, not strings, mapping" in caplog.text

    results = task.total_entailment(["contradiction"])
    assert results == dict(contradiction=1, neutral=0, entailment=0)
    assert isinstance(results, dict)

    results = task.total_entailment([0, 1, 1, "neutral", "entailment", "entailment"])
    assert results == dict(contradiction=1, neutral=3, entailment=2)
    assert sum(results.values()) == 6


def test_tpr_fpr():
    actual_conditions = [0] * 10 + [1] * 10
    predicted_conditions = [0] * 4 + [1] * 6 + [0] * 3 + [1] * 7
    results = task.tpr_fpr(actual_conditions, predicted_conditions)
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

    results = task.tpr_fpr([], [])
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
        task.tpr_fpr(actual_conditions, [predicted_conditions])
    for array in 0, [[0, 1], [1, 0]]:
        with pytest.raises(ValueError):
            task.tpr_fpr(array, array)


def test_abstains():
    y = [0, 1, 2, 3, 4]
    y_pred = np.zeros((5, 10))
    y_pred[1, 1] = 1.0
    assert task.abstains(y, y_pred) == [1, 0, 1, 1, 1]
    for wrong_dim in np.zeros(5), np.zeros((5, 10, 10)):
        with pytest.raises(ValueError):
            task.abstains(y, np.zeros(5))


def test_categorical_accuracy():
    y = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 2, 3]
    assert (task.batch.categorical_accuracy(y, y_pred) == [1, 1, 1, 0, 0]).all()
    assert (task.batch.categorical_accuracy(y, np.eye(5)) == [1] * 5).all()
    with pytest.raises(ValueError):
        task.categorical_accuracy(y, [[y_pred]])


def test_top_n_categorical_accuracy():
    y = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 2, 3]
    assert (
        task.batch.categorical_accuracy(y, y_pred)
        == task.batch.top_n_categorical_accuracy(y, y_pred, n=1)
    ).all()
    y = [2, 0]
    y_pred = [[0.1, 0.4, 0.2, 0.2, 0.1, 0.1], [0.1, 0.4, 0.2, 0.2, 0.1, 0.1]]
    assert (task.batch.top_n_categorical_accuracy(y, y_pred, n=3) == [1, 0]).all()


def test_mAP():
    labels = {"labels": np.array([2]), "boxes": np.array([[0.1, 0.1, 0.7, 0.7]])}

    preds = {
        "labels": np.array([2, 9]),
        "boxes": np.array([[0.1, 0.1, 0.7, 0.7], [0.5, 0.4, 0.9, 0.9]]),
        "scores": np.array([0.8, 0.8]),
    }

    ap_per_class = task.object_detection_AP_per_class([labels], [preds])
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
    ) = task._object_detection_get_tpr_mr_dr_hr(
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
    # Recall that first box isn't counted towards task. Second box has IoU of 1.0,
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

    mean_iou = task.video_tracking_mean_iou(y, y_pred)
    mean_success = task.video_tracking_mean_success_rate(y, y_pred)

    for result in [mean_success, mean_iou]:
        assert isinstance(result, list)
        assert len(result) == len(y)
        assert result[0] == 0.5


def test_word_error_rate():  # and total
    for y, y_pred, errors, length in [
        ("I am here", "no you aren't", 3, 3),
        ("hello", "call me danny", 3, 1),
        ("cat in the hat", "in the cat hat", 2, 4),
        ("no errors", "no errors", 0, 2),
    ]:
        for y_i in (y, bytes(y, encoding="utf-8")):
            for y_pred_i in (y_pred, bytes(y_pred, encoding="utf-8")):
                numer, denom = task.word_error_rate(y_i, y_pred_i)
                assert numer == errors, f"y = {y_i}, y_pred = {y_pred_i}"
                assert denom == length, f"y = {y_i}, y_pred = {y_pred_i}"

    with pytest.raises(TypeError):
        task.word_error_rate(3, "hi there")
    with pytest.raises(TypeError):
        task.word_error_rate("hi there", 3)

    with pytest.raises(TypeError):
        task.total_wer([1, (3, 4)])
    for err in [(1, 2, 3)], [(1,)]:
        with pytest.raises(ValueError):
            task.total_wer(err)

    global_wer, (total_edit_distance, total_words) = task.total_wer([])
    assert math.isnan(global_wer)
    assert (total_edit_distance, total_words) == (0, 0)

    assert (0.8, (8, 10)) == task.total_wer([(3, 3), (3, 1), (2, 4), (0, 2)])


def test_identity():
    x = np.arange(10)
    y = x + 3
    x1, y1 = task.identity_zip(task.identity_unzip(x, y))
    assert (x == x1).all()
    assert (y == y1).all()

    unzipped = []
    stride = 3
    for i in range(0, len(x), stride):
        unzipped.extend(task.identity_unzip(x[i : i + stride], y[i : i + stride]))
    x1, y1 = task.identity_zip(unzipped)
    assert (x == x1).all()
    assert (y == y1).all()


def test_per_class_accuracy():  # and mean
    y = [0, 0, 0, 0, 1, 1, 2, 3, 3, 3]
    y_pred = np.zeros((len(y), 5))
    y_pred[(range(len(y)), [0, 1, 1, 2, 1, 0, 2, 0, 0, 0])] = 1

    results = task.per_class_accuracy(y, y_pred)
    target = {
        0: [1, 0, 0, 0],
        1: [1, 0],
        2: [1],
        3: [0, 0, 0],
        4: [],
    }
    assert target.keys() == results.keys()
    for k in target:
        assert (target[k] == results[k]).all()

    results = task.per_class_mean_accuracy(y, y_pred)
    target = {
        0: 0.25,
        1: 0.5,
        2: 1.0,
        3: 0.0,
        4: float("nan"),
    }
    assert target.keys() == results.keys()
    for k, v in target.items():
        if math.isnan(v):
            assert math.isnan(results[k])
        else:
            assert v == results[k]
