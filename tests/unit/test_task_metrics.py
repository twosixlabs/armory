"""
Test cases for Armory metrics.
"""

import math

import numpy as np
import pytest

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
        entailment_100,
        ground_truth_100,
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

    ap_per_class = task.object_detection_AP_per_class([labels], [preds], mean=False)
    assert ap_per_class[9] == 0
    assert ap_per_class[2] >= 0.99

    meta_result = task.object_detection_AP_per_class([labels], [preds])
    assert "mean" in meta_result
    assert "class" in meta_result
    assert meta_result["mean"] >= 0.99 / 2


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


def generate_square(x, y, length=10):
    """
    Helper function for testing TIDE metrics

    returns: x_min, y_min, x_max, y_max of a square
    """
    return x, y, x + length, y + length


def generate_square_from_iou(square, iou, x3, y3_le_y1=True):
    """
    Helper function for testing TIDE metrics. Given a square (x1, y1, x2, y2), an iou and x3,
    determine y3 to generate another square with the same dimensions.
    Assume x_min, y_min, x_max, y_max format for square

    returns: x3, y3, x4, y4 if new square is possible - returns None otherwise
    """

    x1, y1, x2, y2 = square
    length = x2 - x1  # side of a square
    A1 = A2 = length**2  # restrict to squares of the same size

    if y3_le_y1:
        s_y = 1
    else:
        s_y = -1

    # in this restricted problem, delta_x needs to satisfy certain constraints
    delta_x = abs(x1 - x3)
    # max_delta_x = length * (1 - iou) / (1 + iou)
    # print(
    #     f"delta_x <= {np.round(max_delta_x, decimals=2)}: {delta_x <= max_delta_x}"
    # )
    # # if False, then should return None

    y3 = y1 + s_y * (iou * (A1 + A2) / (1 + iou) / (length - delta_x) - length)

    if (not y3_le_y1 and y3 > y1) or (y3_le_y1 and y3 <= y1):
        return x3, y3, x3 + length, y3 + length
    else:
        return None


def calculate_iou(s1, s2):
    """
    Helper function for testing TIDE metrics. Calculate IoU of two rectangles

    returns: IoU, area of first rectangle, area of second rectangle, width of intersection, height of intersection, area of intersection
    """

    x1, y1, x2, y2 = s1
    x3, y3, x4, y4 = s2

    A1 = abs(x2 - x1) * abs(y2 - y1)
    A2 = abs(x4 - x3) * abs(y4 - y3)

    I_w = max(0, min(max(x1, x2), max(x3, x4)) - max(min(x1, x2), min(x3, x4)))
    I_h = max(0, min(max(y1, y2), max(y3, y4)) - max(min(y1, y2), min(y3, y4)))
    I_A = I_w * I_h
    IoU = I_A / (A1 + A2 - I_A)

    return IoU, A1, A2, I_w, I_h, I_A


def test_tide_metrics():

    x1 = y1 = 10

    s1 = generate_square(x1, y1)
    s_Cls = generate_square_from_iou(s1, 0.8, x1 - 0.5, False)
    # print(s_Cls, calculate_iou(s1, s_Cls))
    s_Loc = generate_square_from_iou(s1, 0.2, x1 - 4.5)
    # print(s_Loc, calculate_iou(s1, s_Loc))
    s_Bkg = generate_square_from_iou(s1, 0.05, x1 + 6.8)
    # print(s_Bkg, calculate_iou(s1, s_Bkg))

    x2 = 35
    s2 = generate_square(x2, y1)
    s_detected = generate_square_from_iou(s2, 0.8, x2 - 0.5, False)
    # print(s_detected, calculate_iou(s2, s_detected))
    s_Dupe = generate_square_from_iou(s2, 0.55, x2 + 2)
    # print(s_Dupe, calculate_iou(s2, s_Dupe))
    s_Both = generate_square_from_iou(s2, 0.2, x2 - 4.5)
    # print(s_Both, calculate_iou(s2, s_Both))

    x3 = 60
    s3 = generate_square(x3, y1)

    y_list = [
        {
            "labels": np.array([1, 2, 3]),
            "boxes": np.array(
                [
                    s1,
                    s2,
                    s3,
                ]
            ),
        }
    ]

    y_pred_list = [
        {
            "labels": np.array([2, 1, 1, 2, 2, 1]),
            "boxes": np.array(
                [
                    s_Cls,
                    s_Loc,
                    s_Bkg,
                    s_detected,
                    s_Dupe,
                    s_Both,
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8]),
        }
    ]

    y_pred_list_Cls = [
        {
            "labels": np.array([1, 1, 1, 2, 2, 1]),
            "boxes": np.array(
                [
                    s_Cls,
                    s_Loc,
                    s_Bkg,
                    s_detected,
                    s_Dupe,
                    s_Both,
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8]),
        }
    ]

    y_pred_list_Loc = [
        {
            "labels": np.array([2, 1, 1, 2, 2, 1]),
            "boxes": np.array(
                [
                    s_Cls,
                    s1,
                    s_Bkg,
                    s_detected,
                    s_Dupe,
                    s_Both,
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8]),
        }
    ]

    y_pred_list_Both = [
        {
            "labels": np.array([2, 1, 1, 2, 2]),
            "boxes": np.array(
                [
                    s_Cls,
                    s_Loc,
                    s_Bkg,
                    s_detected,
                    s_Dupe,
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8, 0.8, 0.8]),
        }
    ]

    y_pred_list_Dupe = [
        {
            "labels": np.array([2, 1, 1, 2, 1]),
            "boxes": np.array(
                [
                    s_Cls,
                    s_Loc,
                    s_Bkg,
                    s_detected,
                    s_Both,
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8, 0.8, 0.8]),
        }
    ]

    y_pred_list_Bkg = [
        {
            "labels": np.array([2, 1, 2, 2, 1]),
            "boxes": np.array(
                [
                    s_Cls,
                    s_Loc,
                    s_detected,
                    s_Dupe,
                    s_Both,
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8, 0.8, 0.8]),
        }
    ]

    y_list_Miss = [
        {
            "labels": np.array([1, 2]),
            "boxes": np.array(
                [
                    s1,
                    s2,
                ]
            ),
        }
    ]

    y_pred_list_All = [
        {
            "labels": np.array([1, 2]),
            "boxes": np.array(
                [
                    s1,
                    s_detected,
                ]
            ),
            "scores": np.array([0.8, 0.8]),
        }
    ]

    results = task.object_detection_mAP_tide(y_list, y_pred_list)
    results_Cls = task.object_detection_mAP_tide(y_list, y_pred_list_Cls)
    results_Loc = task.object_detection_mAP_tide(y_list, y_pred_list_Loc)
    results_Both = task.object_detection_mAP_tide(y_list, y_pred_list_Both)
    results_Dupe = task.object_detection_mAP_tide(y_list, y_pred_list_Dupe)
    results_Bkg = task.object_detection_mAP_tide(y_list, y_pred_list_Bkg)
    results_Miss = task.object_detection_mAP_tide(y_list_Miss, y_pred_list)
    results_All = task.object_detection_mAP_tide(y_list_Miss, y_pred_list_All)

    error_key_list = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]

    def check_assertion(
        armory_output, test_prompt, error_key_list=error_key_list, fixed_key_list=[]
    ):
        for key in error_key_list:
            assert (
                key in armory_output["errors"]["main"]["count"]
            ), f"{test_prompt}: {key} not in results"
            error_count = armory_output["errors"]["main"]["count"][key]
            correct_count = 0 if key in fixed_key_list else 1
            assert (
                error_count == correct_count
            ), f"{test_prompt}: Count for {key} error is not {correct_count}, but {error_count}"

    test_prompt = "Checking TIDE metrics for case with one example of each error type"
    check_assertion(results, test_prompt)

    test_prompt = "Checking TIDE metrics after fixing classification error"
    check_assertion(results_Cls, test_prompt, fixed_key_list=["Cls"])

    test_prompt = "Checking TIDE metrics after fixing localization error"
    check_assertion(results_Loc, test_prompt, fixed_key_list=["Loc"])

    test_prompt = (
        "Checking TIDE metrics after fixing classification and localization error"
    )
    check_assertion(results_Both, test_prompt, fixed_key_list=["Both"])

    test_prompt = "Checking TIDE metrics after fixing duplicate error"
    check_assertion(results_Dupe, test_prompt, fixed_key_list=["Dupe"])

    test_prompt = "Checking TIDE metrics after fixing background error"
    check_assertion(results_Bkg, test_prompt, fixed_key_list=["Bkg"])

    test_prompt = "Checking TIDE metrics after fixing missed error"
    check_assertion(results_Miss, test_prompt, fixed_key_list=["Miss"])

    test_prompt = "Checking TIDE metrics after fixing all errors"
    check_assertion(results_All, test_prompt, fixed_key_list=error_key_list)


def test_tide_metrics_no_overlap():

    x1 = 10
    y1 = 35

    s0 = generate_square(x1, y1)
    s_Cls = generate_square_from_iou(s0, 0.8, x1 - 0.5)
    # print(s_Cls, calculate_iou(s0, s_Cls))

    y2 = y1 - 25
    s1 = generate_square(x1, y2)
    s_Loc = generate_square_from_iou(s1, 0.2, x1 - 4.5)
    # print(s_Loc, calculate_iou(s1, s_Loc))

    x2 = x1 + 25
    s2 = generate_square(x2, y1)
    s_Both = generate_square_from_iou(s2, 0.2, x2 - 4.5)
    # print(s_Both, calculate_iou(s2, s_Both))

    s3 = generate_square(x2, y2)
    s_Bkg = generate_square_from_iou(s3, 0.05, x2 - 6.8)
    # print(s_Bkg, calculate_iou(s3, s_Bkg))

    x3 = x2 + 25
    s4 = generate_square(x3, y1)
    s_detected = generate_square_from_iou(s4, 0.8, x3 + 0.5, False)
    # print(s_detected, calculate_iou(s4, s_detected))
    s_Dupe = generate_square_from_iou(s4, 0.55, x3 - 2)
    # print(s_Dupe, calculate_iou(s4, s_Dupe))

    s5 = generate_square(x3, y2)

    y_list = [
        {
            "labels": np.arange(6),
            "boxes": np.array(
                [
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                ]
            ),
        }
    ]

    y_pred_list = [
        {
            "labels": np.array([1, 1, 3, 3, 4, 4]),
            "boxes": np.array(
                [
                    s_Cls,
                    s_Loc,
                    s_Both,
                    s_Bkg,
                    s_detected,
                    s_Dupe,
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8]),
        }
    ]

    y_pred_list_Cls = [
        {
            "labels": np.array([0, 1, 3, 3, 4, 4]),
            "boxes": np.array(
                [
                    s_Cls,
                    s_Loc,
                    s_Both,
                    s_Bkg,
                    s_detected,
                    s_Dupe,
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8]),
        }
    ]

    y_pred_list_Loc = [
        {
            "labels": np.array([1, 1, 3, 3, 4, 4]),
            "boxes": np.array(
                [
                    s_Cls,
                    s1,
                    s_Both,
                    s_Bkg,
                    s_detected,
                    s_Dupe,
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8]),
        }
    ]

    y_pred_list_Both = [
        {
            "labels": np.array([1, 1, 3, 4, 4]),
            "boxes": np.array(
                [
                    s_Cls,
                    s_Loc,
                    s_Bkg,
                    s_detected,
                    s_Dupe,
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8, 0.8, 0.8]),
        }
    ]

    y_pred_list_Dupe = [
        {
            "labels": np.array([1, 1, 3, 3, 4]),
            "boxes": np.array(
                [
                    s_Cls,
                    s_Loc,
                    s_Both,
                    s_Bkg,
                    s_detected,
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8, 0.8, 0.8]),
        }
    ]

    y_pred_list_Bkg = [
        {
            "labels": np.array([1, 1, 3, 4, 4]),
            "boxes": np.array(
                [
                    s_Cls,
                    s_Loc,
                    s_Both,
                    s_detected,
                    s_Dupe,
                ]
            ),
            "scores": np.array([0.8, 0.8, 0.8, 0.8, 0.8]),
        }
    ]

    y_list_Miss = [
        {
            "labels": np.array([0, 1, 4]),
            "boxes": np.array(
                [
                    s0,
                    s1,
                    s4,
                ]
            ),
        }
    ]

    results = task.object_detection_mAP_tide(y_list, y_pred_list)
    results_Cls = task.object_detection_mAP_tide(y_list, y_pred_list_Cls)
    results_Loc = task.object_detection_mAP_tide(y_list, y_pred_list_Loc)
    results_Both = task.object_detection_mAP_tide(y_list, y_pred_list_Both)
    results_Dupe = task.object_detection_mAP_tide(y_list, y_pred_list_Dupe)
    results_Bkg = task.object_detection_mAP_tide(y_list, y_pred_list_Bkg)
    results_Miss = task.object_detection_mAP_tide(y_list_Miss, y_pred_list)

    base_case_error_count = {
        "Cls": 1,
        "Loc": 1,
        "Both": 1,
        "Dupe": 1,
        "Bkg": 1,
        "Miss": 3,
    }
    fix_Cls_error_count = {
        "Cls": 0,
        "Loc": 1,
        "Both": 1,
        "Dupe": 1,
        "Bkg": 1,
        "Miss": 3,
    }
    fix_Loc_error_count = {
        "Cls": 1,
        "Loc": 0,
        "Both": 1,
        "Dupe": 1,
        "Bkg": 1,
        "Miss": 3,
    }
    fix_Both_error_count = {
        "Cls": 1,
        "Loc": 1,
        "Both": 0,
        "Dupe": 1,
        "Bkg": 1,
        "Miss": 3,
    }
    fix_Dupe_error_count = {
        "Cls": 1,
        "Loc": 1,
        "Both": 1,
        "Dupe": 0,
        "Bkg": 1,
        "Miss": 3,
    }
    fix_Bkg_error_count = {
        "Cls": 1,
        "Loc": 1,
        "Both": 1,
        "Dupe": 1,
        "Bkg": 0,
        "Miss": 3,
    }
    fix_Miss_error_count = {
        "Cls": 1,
        "Loc": 1,
        "Both": 0,
        "Dupe": 1,
        "Bkg": 2,
        "Miss": 0,
    }

    def check_assertion(armory_output, test_prompt, error_count):
        for key, value in error_count.items():
            assert (
                key in armory_output["errors"]["main"]["count"]
            ), f"{test_prompt}: {key} not in results"
            armory_error_count = armory_output["errors"]["main"]["count"][key]
            assert (
                armory_error_count == value
            ), f"{test_prompt}: Count for {key} error is not {value}, but {armory_error_count}"

    test_prompt = "Checking TIDE metrics for case with one example of each error type"
    check_assertion(results, test_prompt, base_case_error_count)

    test_prompt = "Checking TIDE metrics after fixing classification error"
    check_assertion(results_Cls, test_prompt, fix_Cls_error_count)

    test_prompt = "Checking TIDE metrics after fixing localization error"
    check_assertion(results_Loc, test_prompt, fix_Loc_error_count)

    test_prompt = (
        "Checking TIDE metrics after fixing classification and localization error"
    )
    check_assertion(results_Both, test_prompt, fix_Both_error_count)

    test_prompt = "Checking TIDE metrics after fixing duplicate error"
    check_assertion(results_Dupe, test_prompt, fix_Dupe_error_count)

    test_prompt = "Checking TIDE metrics after fixing background error"
    check_assertion(results_Bkg, test_prompt, fix_Bkg_error_count)

    test_prompt = "Checking TIDE metrics after fixing missed error"
    check_assertion(results_Miss, test_prompt, fix_Miss_error_count)
