"""
Metrics for scenarios

Outputs are lists of python variables amenable to JSON serialization:
    e.g., bool, int, float
    numpy data types and tensors generally fail to serialize
"""

from collections import Counter
import os
from typing import Dict, List

import numpy as np
from scipy import stats

from armory import paths
from armory.data.adversarial.apricot_metadata import (
    ADV_PATCH_MAGIC_NUMBER_LABEL_ID,
    APRICOT_PATCHES,
)
from armory.logs import log
from armory.metrics import perturbation
from armory.utils.external_repo import ExternalPipInstalledImport

_ENTAILMENT_MODEL = None


class Entailment:
    """
    Entailment measures the relationship between the premise y and hypothesis y_pred
    It can be categorized as three values:
        0 - "contradiction" - the hypothesis contradicts the premise
        1 - "neutral" - the hypothesis is logically unrelated to the premise
        2 - "entailment" - the hypothesis follows from the premise

    See: https://towardsdatascience.com/fine-tuning-pre-trained-transformer-models-for-sentence-entailment-d87caf9ec9db
    """

    def __init__(self, model_name="roberta-large-mnli", cache_dir=None):
        # Don't generate multiple entailment models
        global _ENTAILMENT_MODEL
        if _ENTAILMENT_MODEL is not None:
            if model_name == _ENTAILMENT_MODEL[0]:
                log.info(f"Using existing entailment model {model_name} for metric")
                self.tokenizer, self.model, self.label_mapping = _ENTAILMENT_MODEL[2:]
                return
            else:
                log.warning(
                    f"Creating new entailment model {model_name} though {_ENTAILMENT_MODEL[0]} exists"
                )

        if cache_dir is None:
            cache_dir = os.path.join(
                paths.runtime_paths().saved_model_dir, "huggingface"
            )

        with ExternalPipInstalledImport(
            package="transformers",
            dockerimage="twosixarmory/pytorch-deepspeech",
        ):
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model.eval()
        self.label_mapping = ["contradiction", "neutral", "entailment"]
        _ENTAILMENT_MODEL = (
            model_name,
            cache_dir,
            self.tokenizer,
            self.model,
            self.label_mapping,
        )

    def __call__(self, y, y_pred):
        import torch

        # In Armory, y is stored as byte strings, and y_pred is stored as strings
        y = np.array(
            [y_i.decode("utf-8") if isinstance(y_i, bytes) else y_i for y_i in y]
        )
        sentence_pairs = list(zip(y, y_pred))
        features = self.tokenizer(
            sentence_pairs, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            scores = self.model(**features).logits
            labels = [self.label_mapping[i] for i in scores.argmax(dim=1)]

        return labels  # return list of labels, not (0, 1, 2)


def total_entailment(sample_results):
    """
    Aggregate a list of per-sample entailment results in ['contradiction', 'neutral', 'entailment'] format
        Return a dictionary of counts for each label
    """
    entailment_map = ["contradiction", "neutral", "entailment"]
    for i in range(len(sample_results)):
        sample = sample_results[i]
        if sample in (0, 1, 2):
            log.warning("Entailment outputs are (0, 1, 2) ints, not strings, mapping")
            sample_results[i] = entailment_map[sample]
        elif sample not in entailment_map:
            raise ValueError(f"result {sample} not a valid entailment label")

    c = Counter()
    c.update(sample_results)
    c = dict(c)  # ensure JSON-able
    for k in entailment_map:
        if k not in c:
            c[k] = 0
    return c


def total_wer(sample_wers):
    """
    Aggregate a list of per-sample word error rate tuples (edit_distance, words)
        Return global_wer, (total_edit_distance, total_words)
    """
    # checks if all values are tuples from the WER metric
    if all(isinstance(wer_tuple, tuple) for wer_tuple in sample_wers):
        total_edit_distance = 0
        total_words = 0
        for wer_tuple in sample_wers:
            total_edit_distance += int(wer_tuple[0])
            total_words += int(wer_tuple[1])
        if total_words:
            global_wer = float(total_edit_distance / total_words)
        else:
            global_wer = float("nan")
        return global_wer, (total_edit_distance, total_words)
    else:
        raise ValueError("total_wer() only for WER metric aggregation")


def identity_unzip(*args):
    """
    Map batchwise args to a list of sample-wise args
    """
    return list(zip(*args))


def identity_zip(sample_arg_tuples):
    args = [list(x) for x in zip(*sample_arg_tuples)]
    return args


# NOTE: Not registered as a metric due to arg in __init__
class MeanAP:
    def __init__(self, ap_metric):
        self.ap_metric = ap_metric

    def __call__(self, values, **kwargs):
        args = [list(x) for x in zip(*values)]
        ap = self.ap_metric(*args, **kwargs)
        mean_ap = np.fromiter(ap.values(), dtype=float).mean()
        return {"mean": mean_ap, "class": ap}


def tpr_fpr(actual_conditions, predicted_conditions):
    """
    actual_conditions and predicted_conditions should be equal length boolean np arrays

    Returns a dict containing TP, FP, TN, FN, TPR, FPR, TNR, FNR, F1 Score
    """
    actual_conditions, predicted_conditions = [
        np.asarray(x, dtype=np.bool) for x in (actual_conditions, predicted_conditions)
    ]
    if actual_conditions.shape != predicted_conditions.shape:
        raise ValueError(
            f"inputs must have equal shape. {actual_conditions.shape} != {predicted_conditions.shape}"
        )
    if actual_conditions.ndim != 1:
        raise ValueError(f"inputs must be 1-dimensional, not {actual_conditions.ndim}")

    true_positives = int(np.sum(predicted_conditions & actual_conditions))
    true_negatives = int(np.sum(~predicted_conditions & ~actual_conditions))
    false_positives = int(np.sum(predicted_conditions & ~actual_conditions))
    false_negatives = int(np.sum(~predicted_conditions & actual_conditions))

    actual_positives = true_positives + false_negatives
    if actual_positives > 0:
        true_positive_rate = true_positives / actual_positives
        false_negative_rate = false_negatives / actual_positives
    else:
        true_positive_rate = false_negative_rate = float("nan")

    actual_negatives = true_negatives + false_positives
    if actual_negatives > 0:
        false_positive_rate = false_positives / actual_negatives
        true_negative_rate = true_negatives / actual_negatives
    else:
        false_positive_rate = true_negative_rate = float("nan")

    if true_positives or false_positives or false_negatives:
        f1_score = true_positives / (
            true_positives + 0.5 * (false_positives + false_negatives)
        )
    else:
        f1_score = float("nan")

    return dict(
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        true_negative_rate=true_negative_rate,
        f1_score=f1_score,
    )


def per_class_accuracy(y, y_pred):
    """
    Return a dict mapping class indices to their accuracies
        Returns nan for classes that are not present

    y - 1-dimensional array
    y_pred - 2-dimensional array
    """
    y, y_pred = (np.asarray(i) for i in (y, y_pred))
    if y.ndim != 1:
        raise ValueError("y should be a list of classes")
    if y.ndim + 1 != y_pred.ndim:
        raise ValueError("y_pred should be a matrix of probabilities")

    results = {}
    for i in range(y_pred.shape[1]):
        if i in y:
            index = y == i
            results[i] = categorical_accuracy(y[index], y_pred[index])
        else:
            results[i] = float("nan")

    return results


def per_class_mean_accuracy(y, y_pred):
    """
    Return a dict mapping class indices to their mean accuracies
        Returns nan for classes that are not present

    y - 1-dimensional array
    y_pred - 2-dimensional array
    """
    return {k: np.mean(v) for k, v in per_class_accuracy(y, y_pred).items()}


def abstains(y, y_pred):
    """
    For each sample in y_pred:
        return 1 for i if y_pred[i] is all 0s (an abstention), return 0 otherwise
        returns a list of (0, 1) elements
    """
    del y
    y_pred = np.asarray(y_pred)
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred {y_pred} is not 2-dimensional")
    return [int(x) for x in (y_pred == 0.0).all(axis=1)]


def compute_chi2_p_value(contingency_table: np.ndarray) -> List[float]:
    """
    Given a 2-x-2 contingency table of the form

                          not flagged by B   |     flagged by B
                      ---------------------------------------------
    not flagged by A |           a           |          b         |
                     |---------------------------------------------
        flagged by A |           c           |          d         |
                      ---------------------------------------------

    perform a chi-squared test to measure the association between
    the A flags and B flags, returning a p-value.
    """
    try:
        _, chi2_p_value, _, _ = stats.chi2_contingency(
            contingency_table, correction=False
        )
    except ValueError:
        chi2_p_value = np.nan
    return [chi2_p_value]


def compute_fisher_p_value(contingency_table: np.ndarray) -> List[float]:
    """
    Given a 2-x-2 contingency table of the form

                          not flagged by B   |     flagged by B
                      ---------------------------------------------
    not flagged by A |           a           |          b         |
                     |---------------------------------------------
        flagged by A |           c           |          d         |
                      ---------------------------------------------

    perform a Fisher exact test to measure the association between
    the A flags and B flags, returning a p-value.
    """
    _, fisher_p_value = stats.fisher_exact(contingency_table, alternative="greater")
    return [fisher_p_value]


def compute_spd(contingency_table: np.ndarray) -> List[float]:
    """
    Given a 2-x-2 contingency table of the form

                          not flagged by B   |     flagged by B
                      ---------------------------------------------
    not flagged by A |           a           |          b         |
                     |---------------------------------------------
        flagged by A |           c           |          d         |
                      ---------------------------------------------

    the Statistical Parity Difference computed by

    SPD = b / (a + b) - d / (c + d)

    is one measure of the impact being flagged by A has on being flagged by B.
    """
    numerators = contingency_table[:, 1]
    denominators = contingency_table.sum(1)
    numerators[denominators == 0] = 0  # Handle division by zero:
    denominators[denominators == 0] = 1  # 0/0 => 0/1.
    fractions = numerators / denominators
    spd = fractions[0] - fractions[1]
    return [spd]


def make_contingency_tables(
    y: np.ndarray, flagged_A: np.ndarray, flagged_B: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Given a list of class labels and two arbitrary binary flags A and B,
    for each class, produce the following 2-x-2 contingency table:

                             not flagged by B   |     flagged by B
                         ---------------------------------------------
       not flagged by A |           a           |          b         |
                        |---------------------------------------------
           flagged by A |           c           |          d         |
                         ---------------------------------------------

    For example, flag A can be whether this example was classified correctly,
    while flag B reports some other binary characteristic of the data.

    Args:
        y (np.ndarray): The true labels (not necessarily binary)
        flagged_A (np.ndarray): The binary outputs of flag A
        flagged_B (np.ndarray): The binary outputs of flag B

    Returns:
        A map (Dict[int, np.ndarray]) of the per-class contingency tables.
    """

    y = np.array(y).astype(np.int).flatten()
    flagged_A = np.array(flagged_A).astype(np.bool_).flatten()
    flagged_B = np.array(flagged_B).astype(np.bool_).flatten()

    if len(flagged_A) != len(y) or len(flagged_B) != len(y):
        raise ValueError(
            f"Expected arrays y, flagged_A, and flagged_B of the same length: \
            got {len(y)}, {len(flagged_A)}, and {len(flagged_B)}."
        )

    contingency_tables = {}
    for class_id in np.unique(y):

        items_flagged_A = flagged_A[y == class_id]
        items_flagged_B = flagged_B[y == class_id]

        a = (~items_flagged_A & ~items_flagged_B).sum()
        b = (~items_flagged_A & items_flagged_B).sum()
        c = (items_flagged_A & ~items_flagged_B).sum()
        d = (items_flagged_A & items_flagged_B).sum()

        table = np.array([[a, b], [c, d]])
        contingency_tables[class_id] = table

    return contingency_tables


def filter_perplexity_fps_benign(
    y_clean: np.ndarray, poison_index: np.ndarray, poison_prediction: np.ndarray
) -> List[float]:
    """
    Measure one possible aspect of bias by seeing how closely the distribution of false
    positives matches the distribution of unpoisoned data.  The intuition is that bias
    depends not on how much of the poison the filter finds, but only what the filter does
    to the rest of the data.  That is, if it removes clean data, it should do so in
    proportion to class frequency, removing roughly the same fraction of each class.

    In particular, we do not expect that _everything_ removed by the filter should match
    the original class distribution, because we do not expect that the poison was applied
    evenly to the whole dataset in the first place.

    Args:
        y_clean:            The class labels of the original dataset
        poison_index:       The indices of elements in y_clean that were actually poisoned
        poison_prediction:  What the filter thinks is poison.  1 for poison, 0 for clean

    Returns:
        Normalized perplexity from fps to benign, where fps is the class frequency distribution of false positives,
        and benign is the class frequency distribution of the unpoisoned data

    """

    # convert poison_index to binary vector the same length as data
    poison_inds = np.zeros_like(y_clean)
    poison_inds[poison_index.astype(np.int64)] = 1
    # benign is here defined to be the class distribution of the unpoisoned part of the data
    x_benign = y_clean[poison_inds == 0]
    x_benign = np.bincount(x_benign, minlength=max(y_clean))
    x_benign = x_benign / x_benign.sum()
    # fps is false positives: clean data marked as poison by the filter
    fp_inds = (1 - poison_inds) & poison_prediction
    fp_labels = y_clean[fp_inds == 1]
    fps = np.bincount(fp_labels, minlength=max(y_clean))
    if fps.sum() == 0:
        return [1]  # If no FPs, we'll define perplexity to be 1 (unbiased)
    fps = fps / fps.sum()

    return perplexity(fps, x_benign)


def perplexity(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> List[float]:
    """
    Return the normalized p-to-q perplexity.
    """
    kl_div_pq = kl_div(p, q, eps)[0]
    perplexity_pq = np.exp(-kl_div_pq)
    return [perplexity_pq]


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> List[float]:
    """
    Return the Kullback-Leibler divergence from p to q.
    """
    cross_entropy_pq = _cross_entropy(p, q, eps)
    entropy_p = _cross_entropy(p, p, eps)
    kl_div_pq = cross_entropy_pq - entropy_p
    return [kl_div_pq]


def _cross_entropy(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Return the cross entropy from a distribution p to a distribution q.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    if p.ndim > 2 or q.ndim > 2:
        raise ValueError(
            f"Not obvious how to reshape arrays: got shapes {p.shape} and {q.shape}."
        )
    elif (p.ndim == 2 and p.shape[0] > 1) or (q.ndim == 2 and q.shape[0] > 1):
        raise ValueError(
            f"Expected 2-dimensional arrays to have shape (1, *): got shapes \
             {p.shape} and {q.shape}."
        )
    p = p.reshape(-1)
    q = q.reshape(-1)
    if p.shape[0] != q.shape[0]:
        raise ValueError(
            f"Expected arrays of the same length: got lengths {len(p)} and {len(q)}."
        )
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Arrays must both be non-negative.")
    if np.isclose(p.sum(), 0) or np.isclose(q.sum(), 0):
        raise ValueError("Arrays must both be non-zero.")
    if not np.isclose(p.sum(), 1):
        p /= p.sum()
    if not np.isclose(q.sum(), 1):
        q /= q.sum()
    cross_entropy_pq = (-p * np.log(q + eps)).sum()
    return cross_entropy_pq


def categorical_accuracy(y, y_pred):
    """
    Return the categorical accuracy of the predictions
    """
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    if y.ndim == 0:
        y = np.array([y])
        y_pred = np.array([y_pred])

    if y.shape == y_pred.shape:
        return [int(x) for x in list(y == y_pred)]
    elif y.ndim + 1 == y_pred.ndim:
        if y.ndim == 0:
            return [int(y == np.argmax(y_pred, axis=-1))]
        return [int(x) for x in list(y == np.argmax(y_pred, axis=-1))]
    else:
        raise ValueError(f"{y} and {y_pred} have mismatched dimensions")


def top_5_categorical_accuracy(y, y_pred):
    """
    Return the top 5 categorical accuracy of the predictions
    """
    return top_n_categorical_accuracy(y, y_pred, 5)


def top_n_categorical_accuracy(y, y_pred, n):
    if n < 1:
        raise ValueError(f"n must be a positive integer, not {n}")
    n = int(n)
    if n == 1:
        return categorical_accuracy(y, y_pred)
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    if y.ndim == 0:
        y = np.array([y])
        y_pred = np.array([y_pred])

    if len(y) != len(y_pred):
        raise ValueError("y and y_pred are of different length")
    if y.shape == y_pred.shape:
        raise ValueError("Must supply multiple predictions for top 5 accuracy")
    elif y.ndim + 1 == y_pred.ndim:
        y_pred_top5 = np.argsort(y_pred, axis=-1)[:, -n:]
        if y.ndim == 0:
            return [int(y in y_pred_top5)]
        return [int(y[i] in y_pred_top5[i]) for i in range(len(y))]
    else:
        raise ValueError(f"{y} and {y_pred} have mismatched dimensions")


def word_error_rate(y, y_pred):
    """
    Return the word error rate for a batch of transcriptions.
    """
    if len(y) != len(y_pred):
        raise ValueError(f"len(y) {len(y)} != len(y_pred) {len(y_pred)}")
    return [_word_error_rate(y_i, y_pred_i) for (y_i, y_pred_i) in zip(y, y_pred)]


def _word_error_rate(y_i, y_pred_i):
    if isinstance(y_i, str):
        reference = y_i.split()
    elif isinstance(y_i, bytes):
        reference = y_i.decode("utf-8").split()
    else:
        raise TypeError(f"y_i is of type {type(y_i)}, expected string or bytes")
    hypothesis = y_pred_i.split()
    r_length = len(reference)
    h_length = len(hypothesis)
    matrix = np.zeros((r_length + 1, h_length + 1))
    for i in range(r_length + 1):
        for j in range(h_length + 1):
            if i == 0:
                matrix[0][j] = j
            elif j == 0:
                matrix[i][0] = i
    for i in range(1, r_length + 1):
        for j in range(1, h_length + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                substitute = matrix[i - 1][j - 1] + 1
                insertion = matrix[i][j - 1] + 1
                deletion = matrix[i - 1][j] + 1
                matrix[i][j] = min(substitute, insertion, deletion)
    return (matrix[r_length][h_length], r_length)


def _check_object_detection_input(y_list, y_pred_list):
    """
    Helper function to check that the object detection labels and predictions are in
    the expected format and contain the expected fields.

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    """
    if not isinstance(y_pred_list, list):
        raise TypeError("Expected y_pred_list to be a list")

    if not isinstance(y_list, list):
        raise TypeError("Expected y_list to be a list")

    if len(y_list) != len(y_pred_list):
        raise ValueError(
            f"Received {len(y_list)} labels but {len(y_pred_list)} predictions"
        )
    elif len(y_list) == 0:
        raise ValueError("Received no labels or predictions")

    REQUIRED_LABEL_KEYS = ["labels", "boxes"]
    REQUIRED_PRED_KEYS = REQUIRED_LABEL_KEYS + ["scores"]

    for (y, y_pred) in zip(y_list, y_pred_list):
        if not all(key in y for key in REQUIRED_LABEL_KEYS):
            raise ValueError(
                f"y must contain the following keys: {REQUIRED_LABEL_KEYS}. The following keys were found: {y.keys()}"
            )
        elif not all(key in y_pred for key in REQUIRED_PRED_KEYS):
            raise ValueError(
                f"y_pred must contain the following keys: {REQUIRED_PRED_KEYS}. The following keys were found: {y_pred.keys()}"
            )


def _check_video_tracking_input(y, y_pred):
    """
    Helper function to check that video tracking labels and predictions are in
    the expected format.

    y (List[Dict, ...]): list of length equal to number of examples. Each element
                  is a dict with "boxes" key mapping to (N, 4) numpy array. Boxes are
                  expected to be in [x1, y1, x2, y2] format.
    y_pred (List[Dict, ...]): same as above
    """
    for input in [y, y_pred]:
        assert isinstance(input, list)
        for input_dict_i in input:
            assert isinstance(input_dict_i, dict)
            assert "boxes" in input_dict_i
    assert len(y) == len(y_pred)
    for i in range(len(y)):
        y_box_array_shape = y[i]["boxes"].shape
        assert y_box_array_shape[1] == 4
        y_pred_box_array_shape = y_pred[i]["boxes"].shape
        assert y_box_array_shape == y_pred_box_array_shape


def _intersection_over_union(box_1, box_2):
    """
    Assumes each input has shape (4,) and format [y1, x1, y2, x2] or [x1, y1, x2, y2]
    """
    assert box_1[2] >= box_1[0]
    assert box_2[2] >= box_2[0]
    assert box_1[3] >= box_1[1]
    assert box_2[3] >= box_2[1]

    if all(i <= 1.0 for i in box_1[np.where(box_1 > 0)]) ^ all(
        i <= 1.0 for i in box_2[np.where(box_2 > 0)]
    ):
        log.warning("One set of boxes appears to be normalized while the other is not")

    # Determine coordinates of intersection box
    x_left = max(box_1[1], box_2[1])
    x_right = min(box_1[3], box_2[3])
    y_top = max(box_1[0], box_2[0])
    y_bottom = min(box_1[2], box_2[2])

    intersect_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    if intersect_area == 0:
        return 0

    box_1_area = (box_1[3] - box_1[1]) * (box_1[2] - box_1[0])
    box_2_area = (box_2[3] - box_2[1]) * (box_2[2] - box_2[0])

    iou = intersect_area / (box_1_area + box_2_area - intersect_area)
    assert iou >= 0
    assert iou <= 1
    return iou


def video_tracking_mean_iou(y, y_pred):
    """
    Mean IOU between ground-truth and predicted boxes, averaged over all frames for a video.
    This function expects to receive a single video's labels/prediction as input.

    y (np.array): numpy array of shape (num_frames, 4)
    y_pred (List[Dict, ...]): list of length equal to number of examples. Each element
                  is a dict with "boxes" key mapping to (N, 4) numpy array
    """
    _check_video_tracking_input(y, y_pred)
    if len(y_pred) > 1:
        raise ValueError(f"y_pred expected to have length of 1, found {len(y_pred)}.")
    mean_ious = []
    for i in range(len(y)):
        y_pred_boxes = y_pred[i]["boxes"]
        y_boxes = y[i]["boxes"]
        num_frames = y_pred_boxes.shape[0]
        # begin with 2nd box to skip y_init in metric calculation
        mean_ious.append(
            np.array(
                [
                    _intersection_over_union(y_boxes[i], y_pred_boxes[i])
                    for i in range(1, num_frames)
                ]
            ).mean()
        )

    return mean_ious


def video_tracking_mean_success_rate(y, y_pred):
    """
    Mean success rate averaged over all thresholds in {0, 0.05, 0.1, ..., 1.0} and all frames.
    This function expects to receive a single video's labels/prediction as input.

    y (List[Dict, ...]): list of length equal to number of examples. Each element
                  is a dict with "boxes" key mapping to (num_frames, 4) numpy array
    y_pred (List[Dict, ...]): list of length equal to number of examples. Each element
                  is a dict with "boxes" key mapping to (num_frames, 4) numpy array
    """
    _check_video_tracking_input(y, y_pred)
    if len(y_pred) > 1:
        raise ValueError(f"y_pred expected to have length of 1, found {len(y_pred)}.")

    thresholds = np.arange(0, 1.0, 0.05)
    mean_success_rates = (
        []
    )  # initialize list that will have length num_videos, which currently is forced to be 1
    for video_idx in range(len(y_pred)):
        success = np.zeros(len(thresholds))

        # Selecting first element since y_pred is forced to have length 1
        y_pred_boxes = y_pred[video_idx]["boxes"]
        y_boxes = y[video_idx]["boxes"]

        num_frames = y_pred_boxes.shape[0]

        # begin with 2nd frame to skip y_init in metric calculation
        ious = [
            _intersection_over_union(y_boxes[i], y_pred_boxes[i])
            for i in range(1, num_frames)
        ]
        for thresh_idx in range(len(thresholds)):
            success[thresh_idx] = np.sum(ious > thresholds[thresh_idx]) / float(
                num_frames - 1
            )  # subtract by 1 since we ignore first frame
        mean_success_rates.append(success.mean())

    return mean_success_rates


def object_detection_AP_per_class(
    y_list, y_pred_list, iou_threshold=0.5, class_list=None
):
    """
    Mean average precision for object detection. The mAP can be computed by taking the mean
    of the AP's across all classes. This metric is computed over all evaluation samples,
    rather than on a per-sample basis.

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    class_list (list, optional): a list of classes, such that all predictions and ground-truths
        with labels NOT in class_list are to be ignored.

    returns: a dictionary mapping each class to the average precision (AP) for the class.
    """
    _check_object_detection_input(y_list, y_pred_list)

    # Precision will be computed at recall points of 0, 0.1, 0.2, ..., 1
    RECALL_POINTS = np.linspace(0, 1, 11)

    # Converting all boxes to a list of dicts (a list for predicted boxes, and a
    # separate list for ground truth boxes), where each dict corresponds to a box and
    # has the following keys "img_idx", "label", "box", as well as "score" for predicted boxes
    pred_boxes_list = []
    gt_boxes_list = []
    for img_idx, (y, y_pred) in enumerate(zip(y_list, y_pred_list)):
        img_labels = y["labels"].flatten()
        img_boxes = y["boxes"].reshape((-1, 4))
        for gt_box_idx in range(img_labels.flatten().shape[0]):
            label = img_labels[gt_box_idx]
            box = img_boxes[gt_box_idx]
            gt_box_dict = {"img_idx": img_idx, "label": label, "box": box}
            gt_boxes_list.append(gt_box_dict)

        for pred_box_idx in range(y_pred["labels"].flatten().shape[0]):
            pred_label = y_pred["labels"][pred_box_idx]
            pred_box = y_pred["boxes"][pred_box_idx]
            pred_score = y_pred["scores"][pred_box_idx]
            pred_box_dict = {
                "img_idx": img_idx,
                "label": pred_label,
                "box": pred_box,
                "score": pred_score,
            }
            pred_boxes_list.append(pred_box_dict)

    # Union of (1) the set of all true classes and (2) the set of all predicted classes
    set_of_class_ids = set([i["label"] for i in gt_boxes_list]) | set(
        [i["label"] for i in pred_boxes_list]
    )

    if class_list:
        # Filter out classes not in class_list
        set_of_class_ids = set(i for i in set_of_class_ids if i in class_list)

    # Remove the class ID that corresponds to a physical adversarial patch in APRICOT
    # dataset, if present
    set_of_class_ids.discard(ADV_PATCH_MAGIC_NUMBER_LABEL_ID)

    # Initialize dict that will store AP for each class
    average_precisions_by_class = {}

    # Compute AP for each class
    for class_id in set_of_class_ids:

        # Build lists that contain all the predicted/ground-truth boxes with a
        # label of class_id
        class_predicted_boxes = []
        class_gt_boxes = []
        for pred_box in pred_boxes_list:
            if pred_box["label"] == class_id:
                class_predicted_boxes.append(pred_box)
        for gt_box in gt_boxes_list:
            if gt_box["label"] == class_id:
                class_gt_boxes.append(gt_box)

        # Determine how many gt boxes (of class_id) there are in each image
        num_gt_boxes_per_img = Counter([gt["img_idx"] for gt in class_gt_boxes])

        # Initialize dict where we'll keep track of whether a gt box has been matched to a
        # prediction yet. This is necessary because if multiple predicted boxes of class_id
        # overlap with a single gt box, only one of the predicted boxes can be considered a
        # true positive
        img_idx_to_gtboxismatched_array = {}
        for img_idx, num_gt_boxes in num_gt_boxes_per_img.items():
            img_idx_to_gtboxismatched_array[img_idx] = np.zeros(num_gt_boxes)

        # Sort all predicted boxes (of class_id) by descending confidence
        class_predicted_boxes.sort(key=lambda x: x["score"], reverse=True)

        # Initialize arrays. Once filled in, true_positives[i] indicates (with a 1 or 0)
        # whether the ith predicted box (of class_id) is a true positive. Likewise for
        # false_positives array
        true_positives = np.zeros(len(class_predicted_boxes))
        false_positives = np.zeros(len(class_predicted_boxes))

        # Iterating over all predicted boxes of class_id
        for pred_idx, pred_box in enumerate(class_predicted_boxes):
            # Only compare gt boxes from the same image as the predicted box
            gt_boxes_from_same_img = [
                gt_box
                for gt_box in class_gt_boxes
                if gt_box["img_idx"] == pred_box["img_idx"]
            ]

            # If there are no gt boxes in the predicted box's image that have the predicted class
            if len(gt_boxes_from_same_img) == 0:
                false_positives[pred_idx] = 1
                continue

            # Iterate over all gt boxes (of class_id) from the same image as the predicted box,
            # determining which gt box has the highest iou with the predicted box
            highest_iou = 0
            for gt_idx, gt_box in enumerate(gt_boxes_from_same_img):
                iou = _intersection_over_union(pred_box["box"], gt_box["box"])
                if iou >= highest_iou:
                    highest_iou = iou
                    highest_iou_gt_idx = gt_idx

            if highest_iou > iou_threshold:
                # If the gt box has not yet been covered
                if (
                    img_idx_to_gtboxismatched_array[pred_box["img_idx"]][
                        highest_iou_gt_idx
                    ]
                    == 0
                ):
                    true_positives[pred_idx] = 1

                    # Record that we've now covered this gt box. Any subsequent
                    # pred boxes that overlap with it are considered false positives
                    img_idx_to_gtboxismatched_array[pred_box["img_idx"]][
                        highest_iou_gt_idx
                    ] = 1
                else:
                    # This gt box was already covered previously (i.e a different predicted
                    # box was deemed a true positive after overlapping with this gt box)
                    false_positives[pred_idx] = 1
            else:
                false_positives[pred_idx] = 1

        # Cumulative sums of false/true positives across all predictions which were sorted by
        # descending confidence
        tp_cumulative_sum = np.cumsum(true_positives)
        fp_cumulative_sum = np.cumsum(false_positives)

        # Total number of gt boxes with a label of class_id
        total_gt_boxes = len(class_gt_boxes)

        if total_gt_boxes > 0:
            recalls = tp_cumulative_sum / total_gt_boxes
        else:
            recalls = np.zeros_like(tp_cumulative_sum)

        precisions = tp_cumulative_sum / (tp_cumulative_sum + fp_cumulative_sum + 1e-8)

        interpolated_precisions = np.zeros(len(RECALL_POINTS))
        # Interpolate the precision at each recall level by taking the max precision for which
        # the corresponding recall exceeds the recall point
        # See http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf
        for i, recall_point in enumerate(RECALL_POINTS):
            precisions_points = precisions[np.where(recalls >= recall_point)]
            # If there's no cutoff at which the recall > recall_point
            if len(precisions_points) == 0:
                interpolated_precisions[i] = 0
            else:
                interpolated_precisions[i] = max(precisions_points)

        # Compute mean precision across the different recall levels
        average_precision = interpolated_precisions.mean()
        average_precisions_by_class[int(class_id)] = np.around(
            average_precision, decimals=2
        )

    return average_precisions_by_class


def object_detection_mAP(y_list, y_pred_list, iou_threshold=0.5, class_list=None):
    """
    Mean average precision for object detection.

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    class_list (list, optional): a list of classes, such that all predictions and ground-truths
        with labels NOT in class_list are to be ignored.

    returns: a scalar value
    """
    ap_per_class = object_detection_AP_per_class(
        y_list, y_pred_list, iou_threshold=iou_threshold, class_list=class_list
    )
    return np.fromiter(ap_per_class.values(), dtype=float).mean()


def _object_detection_get_tpr_mr_dr_hr(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5, class_list=None
):
    """
    Helper function to compute true positive rate, disappearance rate, misclassification rate, and
    hallucinations per image as defined below:

    true positive rate: the percent of ground-truth boxes which are predicted with iou > iou_threshold,
        score > score_threshold, and the correct label
    misclassification rate: the percent of ground-truth boxes which are predicted with iou > iou_threshold,
        score > score_threshold, and the incorrect label
    disappearance rate: 1 - true_positive_rate - misclassification rate
    hallucinations per image: the number of predicted boxes per image that have score > score_threshold and
        iou(predicted_box, ground_truth_box) < iou_threshold for each ground_truth_box


    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    class_list (list, optional): a list of classes, such that all predictions with labels and ground-truths
        with labels NOT in class_list are to be ignored.

    returns: a tuple of length 4 (TPR, MR, DR, HR) where each element is a list of length equal
    to the number of images.
    """

    true_positive_rate_per_img = []
    misclassification_rate_per_img = []
    disappearance_rate_per_img = []
    hallucinations_per_img = []
    for img_idx, (y, y_pred) in enumerate(zip(y_list, y_pred_list)):
        if class_list:
            # Filter out ground-truth classes with labels not in class_list
            indices_to_keep = np.where(np.isin(y["labels"], class_list))
            gt_boxes = y["boxes"][indices_to_keep]
            gt_labels = y["labels"][indices_to_keep]
        else:
            gt_boxes = y["boxes"]
            gt_labels = y["labels"]

        # initialize count of hallucinations
        num_hallucinations = 0
        num_gt_boxes = len(gt_boxes)

        # Initialize arrays that will indicate whether each respective ground-truth
        # box is a true positive or misclassified
        true_positive_array = np.zeros((num_gt_boxes,))
        misclassification_array = np.zeros((num_gt_boxes,))

        # Only consider the model's confident predictions
        conf_pred_indices = np.where(y_pred["scores"] > score_threshold)[0]
        if class_list:
            # Filter out predictions from classes not in class_list kwarg
            conf_pred_indices = conf_pred_indices[
                np.isin(y_pred["labels"][conf_pred_indices], class_list)
            ]

        # For each confident prediction
        for y_pred_idx in conf_pred_indices:
            y_pred_box = y_pred["boxes"][y_pred_idx]

            # Compute the iou between the predicted box and the ground-truth boxes
            ious = np.array([_intersection_over_union(y_pred_box, a) for a in gt_boxes])

            # Determine which ground-truth boxes, if any, the predicted box overlaps with
            overlap_indices = np.where(ious > iou_threshold)[0]

            # If the predicted box doesn't overlap with any ground-truth boxes, increment
            # the hallucination counter and move on to the next predicted box
            if len(overlap_indices) == 0:
                num_hallucinations += 1
                continue

            # For each ground-truth box that the prediction overlaps with
            for y_idx in overlap_indices:
                # If the predicted label is correct, mark that the ground-truth
                # box has a true positive prediction
                if gt_labels[y_idx] == y_pred["labels"][y_pred_idx]:
                    true_positive_array[y_idx] = 1
                else:
                    # Otherwise mark that the ground-truth box has a misclassification
                    misclassification_array[y_idx] = 1

        # Convert these arrays to binary to avoid double-counting (i.e. when multiple
        # predicted boxes overlap with a single ground-truth box)
        true_positive_rate = (true_positive_array > 0).mean()
        misclassification_rate = (misclassification_array > 0).mean()

        # Any ground-truth box that had no overlapping predicted box is considered a
        # disappearance
        disappearance_rate = 1 - true_positive_rate - misclassification_rate

        true_positive_rate_per_img.append(true_positive_rate)
        misclassification_rate_per_img.append(misclassification_rate)
        disappearance_rate_per_img.append(disappearance_rate)
        hallucinations_per_img.append(num_hallucinations)

    return (
        true_positive_rate_per_img,
        misclassification_rate_per_img,
        disappearance_rate_per_img,
        hallucinations_per_img,
    )


def object_detection_true_positive_rate(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5, class_list=None
):
    """
    Computes object detection true positive rate: the percent of ground-truth boxes which
    are predicted with iou > iou_threshold, score > score_threshold, and the correct label.

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    class_list (list, optional): a list of classes, such that all predictions and ground-truths
        with labels NOT in class_list are to be ignored.

    returns: a list of length equal to the number of images.
    """

    _check_object_detection_input(y_list, y_pred_list)
    true_positive_rate_per_img, _, _, _ = _object_detection_get_tpr_mr_dr_hr(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_list=class_list,
    )
    return true_positive_rate_per_img


def object_detection_misclassification_rate(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5, class_list=None
):
    """
    Computes object detection misclassification rate: the percent of ground-truth boxes which
    are predicted with iou > iou_threshold, score > score_threshold, and an incorrect label.

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    class_list (list, optional): a list of classes, such that all predictions and ground-truths
        with labels NOT in class_list are to be ignored.

    returns: a list of length equal to the number of images
    """

    _check_object_detection_input(y_list, y_pred_list)
    _, misclassification_rate_per_image, _, _ = _object_detection_get_tpr_mr_dr_hr(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_list=class_list,
    )
    return misclassification_rate_per_image


def object_detection_disappearance_rate(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5, class_list=None
):
    """
    Computes object detection disappearance rate: the percent of ground-truth boxes for which
    not one predicted box with score > score_threshold has an iou > iou_threshold with the
    ground-truth box.

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    class_list (list, optional): a list of classes, such that all predictions and ground-truths
        with labels NOT in class_list are to be ignored.

    returns: a list of length equal to the number of images
    """

    _check_object_detection_input(y_list, y_pred_list)
    _, _, disappearance_rate_per_img, _ = _object_detection_get_tpr_mr_dr_hr(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_list=class_list,
    )
    return disappearance_rate_per_img


def object_detection_hallucinations_per_image(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5, class_list=None
):
    """
    Computes object detection hallucinations per image: the number of predicted boxes per image
    that have score > score_threshold and an iou < iou_threshold with each ground-truth box.

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    class_list (list, optional): a list of classes, such that all predictions and ground-truths
        with labels NOT in class_list are to be ignored.

    returns: a list of length equal to the number of images
    """

    _check_object_detection_input(y_list, y_pred_list)
    _, _, _, hallucinations_per_image = _object_detection_get_tpr_mr_dr_hr(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_list=class_list,
    )
    return hallucinations_per_image


def carla_od_hallucinations_per_image(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5
):
    """
    CARLA object detection datasets contains class labels 1-4, with class 4 representing
    the green screen/patch itself, which should not be treated as an object class.
    """
    class_list = [1, 2, 3]
    return object_detection_hallucinations_per_image(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_list=class_list,
    )


def carla_od_disappearance_rate(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5
):
    """
    CARLA object detection datasets contains class labels 1-4, with class 4 representing
    the green screen/patch itself, which should not be treated as an object class.
    """
    class_list = [1, 2, 3]
    return object_detection_disappearance_rate(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_list=class_list,
    )


def carla_od_true_positive_rate(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5
):
    """
    CARLA object detection datasets contains class labels 1-4, with class 4 representing
    the green screen/patch itself, which should not be treated as an object class.
    """
    class_list = [1, 2, 3]
    return object_detection_true_positive_rate(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_list=class_list,
    )


def carla_od_misclassification_rate(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5
):
    """
    CARLA object detection datasets contains class labels 1-4, with class 4 representing
    the green screen/patch itself, which should not be treated as an object class.
    """
    class_list = [1, 2, 3]
    return object_detection_misclassification_rate(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_list=class_list,
    )


def carla_od_AP_per_class(y_list, y_pred_list, iou_threshold=0.5):
    class_list = [1, 2, 3]
    """
    CARLA object detection datasets contains class labels 1-4, with class 4 representing
    the green screen/patch itself, which should not be treated as an object class.
    """
    return object_detection_AP_per_class(
        y_list, y_pred_list, iou_threshold=iou_threshold, class_list=class_list
    )


def apricot_patch_targeted_AP_per_class(y_list, y_pred_list, iou_threshold=0.1):
    """
    Average precision indicating how successfully the APRICOT patch causes the detector
    to predict the targeted class of the patch at the location of the patch. A higher
    value for this metric implies a more successful patch.

    The box associated with the patch is assigned the label of the patch's targeted class.
    Thus, a true positive is the case where the detector predicts the patch's targeted
    class (at a location overlapping the patch). A false positive is the case where the
    detector predicts a non-targeted class at a location overlapping the patch. If the
    detector predicts multiple instances of the target class (that overlap with the patch),
    one of the predictions is considered a true positive and the others are ignored.

    This metric is computed over all evaluation samples, rather than on a per-sample basis.
    It returns a dictionary mapping each class to the average precision (AP) for the class.
    The only classes with potentially nonzero AP's are the classes targeted by the patches
    (see above paragraph).

    From https://arxiv.org/abs/1912.08166: use a low IOU since "the patches will sometimes
    generate many small, overlapping predictions in the region of the attack"

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    """
    _check_object_detection_input(y_list, y_pred_list)

    # Precision will be computed at recall points of 0, 0.1, 0.2, ..., 1
    RECALL_POINTS = np.linspace(0, 1, 11)

    # Converting boxes to a list of dicts (a list for predicted boxes that overlap with the patch,
    # and a separate list for ground truth patch boxes), where each dict corresponds to a box and
    # has the following keys "img_idx", "label", "box", as well as "score" for predicted boxes
    patch_boxes_list = []
    overlappping_pred_boxes_list = []
    for img_idx, (y, y_pred) in enumerate(zip(y_list, y_pred_list)):
        idx_of_patch = np.where(
            y["labels"].flatten() == ADV_PATCH_MAGIC_NUMBER_LABEL_ID
        )[0]
        patch_box = y["boxes"].reshape((-1, 4))[idx_of_patch].flatten()
        patch_id = int(y["patch_id"].flatten()[idx_of_patch])
        patch_target_label = APRICOT_PATCHES[patch_id]["adv_target"]
        patch_box_dict = {
            "img_idx": img_idx,
            "label": patch_target_label,
            "box": patch_box,
        }
        patch_boxes_list.append(patch_box_dict)

        for pred_box_idx in range(y_pred["labels"].size):
            box = y_pred["boxes"][pred_box_idx]
            if _intersection_over_union(box, patch_box) > iou_threshold:
                label = y_pred["labels"][pred_box_idx]
                score = y_pred["scores"][pred_box_idx]
                pred_box_dict = {
                    "img_idx": img_idx,
                    "label": label,
                    "box": box,
                    "score": score,
                }
                overlappping_pred_boxes_list.append(pred_box_dict)

    # Union of (1) the set of classes targeted by patches and (2) the set of all classes
    # predicted at a location that overlaps the patch in the image
    set_of_class_ids = set([i["label"] for i in patch_boxes_list]) | set(
        [i["label"] for i in overlappping_pred_boxes_list]
    )

    # Initialize dict that will store AP for each class
    average_precisions_by_class = {}

    # Compute AP for each class
    for class_id in set_of_class_ids:
        # Build lists that contain all the predicted and patch boxes with a
        # label of class_id
        class_predicted_boxes = []
        class_patch_boxes = []
        for pred_box in overlappping_pred_boxes_list:
            if pred_box["label"] == class_id:
                class_predicted_boxes.append(pred_box)
        for patch_box in patch_boxes_list:
            if patch_box["label"] == class_id:
                class_patch_boxes.append(patch_box)

        # Determine how many patch boxes (of class_id) there are in each image
        num_patch_boxes_per_img = Counter([gt["img_idx"] for gt in class_patch_boxes])

        # Initialize dict where we'll keep track of whether a patch box has been matched to a
        # prediction yet. This is necessary because if multiple predicted boxes of class_id
        # overlap with a patch box, only one of the predicted boxes can be considered a
        # true positive. The rest will be ignored
        img_idx_to_patchboxismatched_array = {}
        for img_idx, num_patch_boxes in num_patch_boxes_per_img.items():
            img_idx_to_patchboxismatched_array[img_idx] = np.zeros(num_patch_boxes)

        # Sort all predicted boxes (of class_id) by descending confidence
        class_predicted_boxes.sort(key=lambda x: x["score"], reverse=True)

        # Initialize list. Once filled in, true_positives[i] indicates (with a 1 or 0)
        # whether the ith predicted box (of class_id) is a true positive or false positive
        is_true_positive = []

        # Iterating over all predicted boxes of class_id
        for pred_idx, pred_box in enumerate(class_predicted_boxes):
            # Only compare patch boxes from the same image as the predicted box
            patch_boxes_from_same_img = [
                patch_box
                for patch_box in class_patch_boxes
                if patch_box["img_idx"] == pred_box["img_idx"]
            ]

            # If there are no patch boxes in the predicted box's image that target the predicted class
            if len(patch_boxes_from_same_img) == 0:
                is_true_positive.append(0)
                continue

            # Iterate over all patch boxes (of class_id) from the same image as the predicted box,
            # determining which patch box has the highest iou with the predicted box.
            highest_iou = 0
            for patch_idx, patch_box in enumerate(patch_boxes_from_same_img):
                iou = _intersection_over_union(pred_box["box"], patch_box["box"])
                if iou >= highest_iou:
                    highest_iou = iou
                    highest_iou_patch_idx = patch_idx

            # If the patch box has not yet been covered
            if (
                img_idx_to_patchboxismatched_array[pred_box["img_idx"]][
                    highest_iou_patch_idx
                ]
                == 0
            ):
                is_true_positive.append(1)

                # Record that we've now covered this patch box. Any subsequent
                # pred boxes that overlap with it are ignored
                img_idx_to_patchboxismatched_array[pred_box["img_idx"]][
                    highest_iou_patch_idx
                ] = 1
            else:
                # This patch box was already covered previously (i.e a different predicted
                # box was deemed a true positive after overlapping with this patch box).
                # The predicted box is thus ignored.
                continue

        # Cumulative sums of false/true positives across all predictions which were sorted by
        # descending confidence
        tp_cumulative_sum = np.cumsum(is_true_positive)
        fp_cumulative_sum = np.cumsum([not i for i in is_true_positive])

        # Total number of patch boxes with a label of class_id
        total_patch_boxes = len(class_patch_boxes)

        if total_patch_boxes > 0:
            recalls = tp_cumulative_sum / total_patch_boxes
        else:
            recalls = np.zeros_like(tp_cumulative_sum)

        precisions = tp_cumulative_sum / (tp_cumulative_sum + fp_cumulative_sum + 1e-8)

        interpolated_precisions = np.zeros(len(RECALL_POINTS))
        # Interpolate the precision at each recall level by taking the max precision for which
        # the corresponding recall exceeds the recall point
        # See http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf
        for i, recall_point in enumerate(RECALL_POINTS):
            precisions_points = precisions[np.where(recalls >= recall_point)]
            # If there's no cutoff at which the recall > recall_point
            if len(precisions_points) == 0:
                interpolated_precisions[i] = 0
            else:
                interpolated_precisions[i] = max(precisions_points)

        # Compute mean precision across the different recall levels
        average_precision = interpolated_precisions.mean()
        average_precisions_by_class[int(class_id)] = np.around(
            average_precision, decimals=2
        )

    return average_precisions_by_class


def dapricot_patch_targeted_AP_per_class(y_list, y_pred_list, iou_threshold=0.1):
    """
    Average precision indicating how successfully the patch causes the detector
    to predict the targeted class of the patch at the location of the patch. A higher
    value for this metric implies a more successful patch.

    The box associated with the patch is assigned the label of the patch's targeted class.
    Thus, a true positive is the case where the detector predicts the patch's targeted
    class (at a location overlapping the patch). A false positive is the case where the
    detector predicts a non-targeted class at a location overlapping the patch. If the
    detector predicts multiple instances of the target class (that overlap with the patch),
    one of the predictions is considered a true positive and the others are ignored.

    This metric is computed over all evaluation samples, rather than on a per-sample basis.
    It returns a dictionary mapping each class to the average precision (AP) for the class.
    The only classes with potentially nonzero AP's are the classes targeted by the patches
    (see above paragraph).

    Assumptions made for D-APRICOT dataset: each image has one ground truth box. This box corresponds
    to the patch and is assigned a label of whatever the attack's target label is. There are no
    ground-truth boxes of COCO objects.

    From https://arxiv.org/abs/1912.08166: use a low IOU since "the patches will sometimes
    generate many small, overlapping predictions in the region of the attack"

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.

    """
    _check_object_detection_input(y_list, y_pred_list)

    # Precision will be computed at recall points of 0, 0.1, 0.2, ..., 1
    RECALL_POINTS = np.linspace(0, 1, 11)

    # Converting boxes to a list of dicts (a list for predicted boxes that overlap with the patch,
    # and a separate list for ground truth patch boxes), where each dict corresponds to a box and
    # has the following keys "img_idx", "label", "box", as well as "score" for predicted boxes
    patch_boxes_list = []
    overlappping_pred_boxes_list = []
    for img_idx, (y, y_pred) in enumerate(zip(y_list, y_pred_list)):
        patch_box = y["boxes"].flatten()
        patch_target_label = int(y["labels"])
        patch_box_dict = {
            "img_idx": img_idx,
            "label": patch_target_label,
            "box": patch_box,
        }
        patch_boxes_list.append(patch_box_dict)

        for pred_box_idx in range(y_pred["labels"].size):
            box = y_pred["boxes"][pred_box_idx]
            if _intersection_over_union(box, patch_box) > iou_threshold:
                label = y_pred["labels"][pred_box_idx]
                score = y_pred["scores"][pred_box_idx]
                pred_box_dict = {
                    "img_idx": img_idx,
                    "label": label,
                    "box": box,
                    "score": score,
                }
                overlappping_pred_boxes_list.append(pred_box_dict)

    # Only compute AP of classes targeted by patches. The D-APRICOT dataset in some
    # cases contains unlabeled COCO objects in the background
    set_of_class_ids = set([i["label"] for i in patch_boxes_list])

    # Initialize dict that will store AP for each class
    average_precisions_by_class = {}

    # Compute AP for each class
    for class_id in set_of_class_ids:
        # Build lists that contain all the predicted and patch boxes with a
        # label of class_id
        class_predicted_boxes = []
        class_patch_boxes = []
        for pred_box in overlappping_pred_boxes_list:
            if pred_box["label"] == class_id:
                class_predicted_boxes.append(pred_box)
        for patch_box in patch_boxes_list:
            if patch_box["label"] == class_id:
                class_patch_boxes.append(patch_box)

        # Determine how many patch boxes (of class_id) there are in each image
        num_patch_boxes_per_img = Counter([gt["img_idx"] for gt in class_patch_boxes])

        # Initialize dict where we'll keep track of whether a patch box has been matched to a
        # prediction yet. This is necessary because if multiple predicted boxes of class_id
        # overlap with a patch box, only one of the predicted boxes can be considered a
        # true positive. The rest will be ignored
        img_idx_to_patchboxismatched_array = {}
        for img_idx, num_patch_boxes in num_patch_boxes_per_img.items():
            img_idx_to_patchboxismatched_array[img_idx] = np.zeros(num_patch_boxes)

        # Sort all predicted boxes (of class_id) by descending confidence
        class_predicted_boxes.sort(key=lambda x: x["score"], reverse=True)

        # Initialize list. Once filled in, true_positives[i] indicates (with a 1 or 0)
        # whether the ith predicted box (of class_id) is a true positive or false positive
        is_true_positive = []

        # Iterating over all predicted boxes of class_id
        for pred_idx, pred_box in enumerate(class_predicted_boxes):
            # Only compare patch boxes from the same image as the predicted box
            patch_boxes_from_same_img = [
                patch_box
                for patch_box in class_patch_boxes
                if patch_box["img_idx"] == pred_box["img_idx"]
            ]

            # If there are no patch boxes in the predicted box's image that target the predicted class
            if len(patch_boxes_from_same_img) == 0:
                is_true_positive.append(0)
                continue

            # Iterate over all patch boxes (of class_id) from the same image as the predicted box,
            # determining which patch box has the highest iou with the predicted box.
            highest_iou = 0
            for patch_idx, patch_box in enumerate(patch_boxes_from_same_img):
                iou = _intersection_over_union(pred_box["box"], patch_box["box"])
                if iou >= highest_iou:
                    highest_iou = iou
                    highest_iou_patch_idx = patch_idx

            # If the patch box has not yet been covered
            if (
                img_idx_to_patchboxismatched_array[pred_box["img_idx"]][
                    highest_iou_patch_idx
                ]
                == 0
            ):
                is_true_positive.append(1)

                # Record that we've now covered this patch box. Any subsequent
                # pred boxes that overlap with it are ignored
                img_idx_to_patchboxismatched_array[pred_box["img_idx"]][
                    highest_iou_patch_idx
                ] = 1
            else:
                # This patch box was already covered previously (i.e a different predicted
                # box was deemed a true positive after overlapping with this patch box).
                # The predicted box is thus ignored.
                continue

        # Cumulative sums of false/true positives across all predictions which were sorted by
        # descending confidence
        tp_cumulative_sum = np.cumsum(is_true_positive)
        fp_cumulative_sum = np.cumsum([not i for i in is_true_positive])

        # Total number of patch boxes with a label of class_id
        total_patch_boxes = len(class_patch_boxes)

        if total_patch_boxes > 0:
            recalls = tp_cumulative_sum / total_patch_boxes
        else:
            recalls = np.zeros_like(tp_cumulative_sum)

        precisions = tp_cumulative_sum / (tp_cumulative_sum + fp_cumulative_sum + 1e-8)

        interpolated_precisions = np.zeros(len(RECALL_POINTS))
        # Interpolate the precision at each recall level by taking the max precision for which
        # the corresponding recall exceeds the recall point
        # See http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf
        for i, recall_point in enumerate(RECALL_POINTS):
            precisions_points = precisions[np.where(recalls >= recall_point)]
            # If there's no cutoff at which the recall > recall_point
            if len(precisions_points) == 0:
                interpolated_precisions[i] = 0
            else:
                interpolated_precisions[i] = max(precisions_points)

        # Compute mean precision across the different recall levels
        average_precision = interpolated_precisions.mean()
        average_precisions_by_class[int(class_id)] = np.around(
            average_precision, decimals=2
        )

    return average_precisions_by_class


def dapricot_patch_target_success(
    y_list, y_pred_list, iou_threshold=0.1, conf_threshold=0.5
):
    """
    Binary metric that simply indicates whether or not the model predicted the targeted
    class at the location of the patch (given an IOU threshold which defaults to 0.1) with
    confidence >= a confidence threshold which defaults to 0.5.

    Assumptions made for D-APRICOT dataset: each image has one ground truth box. This box
    corresponds to the patch and is assigned a label of whatever the attack's target label is.
    There are no ground-truth boxes of COCO objects.

    Note: from https://arxiv.org/abs/1912.08166: by default a low IOU threshold is used since
    "the patches will sometimes generate many small, overlapping predictions in the region
    of the attack"

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    """
    return [
        _dapricot_patch_target_success(
            y, y_pred, iou_threshold=iou_threshold, conf_threshold=conf_threshold
        )
        for y, y_pred in zip(y_list, y_pred_list)
    ]


def _dapricot_patch_target_success(y, y_pred, iou_threshold=0.1, conf_threshold=0.5):
    target_label = int(y["labels"])
    target_box = y["boxes"].reshape((4,))
    pred_indices = np.where(y_pred["scores"] > conf_threshold)[0]
    for pred_idx in pred_indices:
        if y_pred["labels"][pred_idx] == target_label:
            if (
                _intersection_over_union(y_pred["boxes"][pred_idx], target_box)
                > iou_threshold
            ):
                return 1
    return 0


SUPPORTED_METRICS = {
    "entailment": Entailment,
    "total_entailment": total_entailment,
    "total_wer": total_wer,
    "identity_unzip": identity_unzip,
    "identity_zip": identity_zip,
    "tpr_fpr": tpr_fpr,
    "per_class_accuracy": per_class_accuracy,
    "per_class_mean_accuracy": per_class_mean_accuracy,
    "dapricot_patch_target_success": dapricot_patch_target_success,
    "dapricot_patch_targeted_AP_per_class": dapricot_patch_targeted_AP_per_class,
    "apricot_patch_targeted_AP_per_class": apricot_patch_targeted_AP_per_class,
    "categorical_accuracy": categorical_accuracy,
    "top_n_categorical_accuracy": top_n_categorical_accuracy,
    "top_5_categorical_accuracy": top_5_categorical_accuracy,
    "video_tracking_mean_iou": video_tracking_mean_iou,
    "video_tracking_mean_success_rate": video_tracking_mean_success_rate,
    "word_error_rate": word_error_rate,
    "object_detection_AP_per_class": object_detection_AP_per_class,
    "object_detection_mAP": object_detection_mAP,
    "object_detection_disappearance_rate": object_detection_disappearance_rate,
    "object_detection_hallucinations_per_image": object_detection_hallucinations_per_image,
    "object_detection_misclassification_rate": object_detection_misclassification_rate,
    "object_detection_true_positive_rate": object_detection_true_positive_rate,
    "carla_od_AP_per_class": carla_od_AP_per_class,
    "carla_od_disappearance_rate": carla_od_disappearance_rate,
    "carla_od_hallucinations_per_image": carla_od_hallucinations_per_image,
    "carla_od_misclassification_rate": carla_od_misclassification_rate,
    "carla_od_true_positive_rate": carla_od_true_positive_rate,
    "kl_div": kl_div,
    "perplexity": perplexity,
    "filter_perplexity_fps_benign": filter_perplexity_fps_benign,
    "poison_chi2_p_value": compute_chi2_p_value,
    "poison_fisher_p_value": compute_fisher_p_value,
    "poison_spd": compute_spd,
}

assert not any(k in perturbation.batch for k in SUPPORTED_METRICS)
SUPPORTED_METRICS.update(perturbation.batch)


def get_supported_metric(name):
    try:
        function = SUPPORTED_METRICS[name]
    except KeyError:
        raise KeyError(f"{name} is not part of armory.utils.metrics")
    if isinstance(function, type) and issubclass(function, object):
        # If a class is given, instantiate it
        function = function()
    assert callable(function), f"function {name} is not callable"
    return function
