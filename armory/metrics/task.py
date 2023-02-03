"""
Task metrics (comparing y to y_pred)
"""

from collections import Counter
import functools
import os

import numpy as np
from tidecv import TIDE
import tidecv.data

from armory import paths
from armory.data.adversarial.apricot_metadata import (
    ADV_PATCH_MAGIC_NUMBER_LABEL_ID,
    APRICOT_PATCHES,
)
from armory.logs import log
from armory.metrics.common import (
    MetricNameSpace,
    as_batch,
    result_formatter,
    set_namespace,
)
from armory.utils.external_repo import ExternalPipInstalledImport

aggregate = MetricNameSpace()
population = MetricNameSpace()
batch = MetricNameSpace()
element = MetricNameSpace()

# Maps batch or element metrics to aggregation functions
AGGREGATION_MAP = {}


def aggregator(metric, name=None):
    """
    Register an aggregate metric
        These are typically used to combine intermediate results into final results
        Examples include total word error rate and mean average precision
    """
    return set_namespace(aggregate, metric, name=name, set_global=True)


def populationwise(metric, name=None):
    """
    Register a population-wise (full test set) metric
        Similar to a batch metric, but requires the entire set of data points
    """
    return set_namespace(population, metric, name=name, set_global=True)


def batchwise(metric, name=None):
    """
    Register a batch-wise metric
    """
    return set_namespace(batch, metric, name=name, set_global=True)


def elementwise(metric, name=None):
    """
    Register a element-wise metric and register a batch-wise version of it
    """
    if name is None:
        name = metric.__name__
    set_namespace(element, metric, name=name)
    batch_metric = as_batch(metric)
    batchwise(batch_metric, name=name)
    return metric


def _to_numpy(array):
    """
    Map to numpy array with `asarray`, but handle ragged arrays
    """
    try:
        return np.asarray(array)
    except ValueError:
        return np.asarray(array, dtype=object)


def map_to_aggregator(name, aggregator):
    global AGGREGATION_MAP
    if name in AGGREGATION_MAP:
        raise ValueError(f"{name} already mapped to {aggregator}")
    AGGREGATION_MAP[name] = aggregator


def get_aggregator_name(name):
    global AGGREGATION_MAP
    return AGGREGATION_MAP.get(name, None)


def numpy(function):
    """
    Ensures args (but not kwargs) are passed in as numpy vectors

    In contrast to perturbation.numpy, this does not ensure same shape or dtype
    """

    @functools.wraps(function)
    def wrapper(y, y_pred, **kwargs):
        y, y_pred = (_to_numpy(array) for array in (y, y_pred))
        return function(y, y_pred, **kwargs)

    return wrapper


_ENTAILMENT_MODEL = None


# batchwise registration after class definition due to name change
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
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


batchwise(Entailment, name="entailment")


@aggregator
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


@aggregator
def safe_mean(results):
    try:
        return np.mean(results)
    except (TypeError, ValueError):
        return "<'np.mean' failed on given results>"


@aggregator
def total_wer(sample_wers):
    """
    Aggregate an array or list of per-sample word error rate [edit_distance, words]
        Return global_wer, (total_edit_distance, total_words)

    sample_wers: a list of 2-tuples, or an array of shape (N, 2)
    """

    if isinstance(sample_wers, list):
        if not all(len(wer_tuple) == 2 for wer_tuple in sample_wers):
            raise ValueError("Inputs must be tuples of size 2: (edit distance, length)")
    elif isinstance(sample_wers, np.ndarray):
        if sample_wers.ndim != 2 or sample_wers.shape[-1] != 2:
            raise ValueError(
                f"sample_wers must be an array of shape (N, 2). Received shape {sample_wers.shape}"
            )
    else:
        raise ValueError(
            f"Expected sample_wers to be a list or numpy array. Received type {type(sample_wers)}"
        )

    total_edit_distance = 0
    total_words = 0
    for edit_distance, words in sample_wers:
        total_edit_distance += int(edit_distance)
        total_words += int(words)

    if total_words:
        global_wer = float(total_edit_distance / total_words)
    else:
        global_wer = float("nan")
    return global_wer, (total_edit_distance, total_words)


@batchwise
def identity_unzip(*args):
    """
    Map batchwise args to a list of sample-wise args
    """
    return list(zip(*args))


@aggregator
def identity_zip(sample_arg_tuples):
    args = [list(x) for x in zip(*sample_arg_tuples)]
    return args


def mean_ap(ap_per_class):
    """
    Takes the mean across classes and returns a dict with mean and class AP
    """
    mean_ap = np.fromiter(ap_per_class.values(), dtype=float).mean()
    return {"mean": mean_ap, "class": ap_per_class}


@populationwise
def tpr_fpr(actual_conditions, predicted_conditions):
    """
    actual_conditions and predicted_conditions should be equal length boolean np arrays

    Returns a dict containing TP, FP, TN, FN, TPR, FPR, TNR, FNR, F1 Score
    """
    actual_conditions, predicted_conditions = [
        np.asarray(x, dtype=bool) for x in (actual_conditions, predicted_conditions)
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


@batchwise
def per_class_accuracy(y, y_pred):
    """
    Return a dict mapping class indices to a list of their accuracies

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
            results[i] = batch.categorical_accuracy(y[index], y_pred[index])
        else:
            results[i] = np.array([])

    return results


@batchwise
def per_class_mean_accuracy(y, y_pred):
    """
    Return a dict mapping class indices to their mean accuracies
        Returns nan for classes that are not present

    y - 1-dimensional array
    y_pred - 2-dimensional array
    """
    return {
        k: np.mean(v) if len(v) else float("nan")
        for k, v in per_class_accuracy(y, y_pred).items()
    }


@batchwise
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


@elementwise
@numpy
def categorical_accuracy(y, y_pred):
    """
    Return the categorical accuracy of the predictions

    y - integer value, one-hot encoding, or similar score
    y_pred - integer value, one-hot encoding, log loss, probability distribution, logit
    """
    if y.ndim > 1 or y_pred.ndim > 1:
        raise ValueError(f"y {y} and y_pred {y_pred} cannot have 2+ dimensions")
    if y.ndim == 1 and y_pred.ndim == 1 and y.shape != y_pred.shape:
        raise ValueError(f"When vectors, y {y} and y {y_pred} must have same shape")

    if y.ndim == 1:
        y = np.argmax(y)
    if y_pred.ndim == 1:
        y_pred = np.argmax(y_pred)
    return float(y == y_pred)


@elementwise
def top_5_categorical_accuracy(y, y_pred):
    """
    Return the top 5 categorical accuracy of the predictions
    """
    return top_n_categorical_accuracy(y, y_pred, n=5)


@elementwise
@numpy
def top_n_categorical_accuracy(y, y_pred, *, n=5):
    """
    Return the top n categorical accuracy of the predictions

    y_pred - must be a vector of values or a single value
    """
    if n < 1 or n == np.inf or n != int(n):
        raise ValueError(f"n must be a positive integer, not {n}")
    n = int(n)

    if y_pred.ndim == 0:
        return float(y == y_pred)
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred {y_pred} must be a 1-dimensional vector")
    if y.ndim > 1:
        raise ValueError(f"y {y} cannot have > 1 dimension")
    if y.ndim == 1 and y.shape != y_pred.shape:
        raise ValueError(f"When vectors, y {y} and y {y_pred} must have same shape")
    if y.ndim == 1:
        y = np.argmax(y)

    y_pred_top_n = np.argsort(y_pred)[-n:]  # ascending order
    return float(y in y_pred_top_n)


@elementwise
def word_error_rate(y, y_pred):
    """
    Return the word error rate for a batch of transcriptions.
    """
    if isinstance(y, bytes):
        y = y.decode("utf-8")
    elif not isinstance(y, str):
        raise TypeError(f"y is of type {type(y)}, expected string or bytes")
    if isinstance(y_pred, bytes):
        y_pred = y_pred.decode("utf-8")
    elif not isinstance(y_pred, str):
        raise TypeError(f"y_pred is of type {type(y_pred)}, expected string or bytes")
    reference = y.split()
    hypothesis = y_pred.split()

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


@batchwise
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


@batchwise
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


@populationwise
def object_detection_AP_per_class(
    y_list,
    y_pred_list,
    iou_threshold=0.5,
    class_list=None,
    mean=True,
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

    mean: if False, returns a dict mapping each class to its average precision (AP)
        if True, calls `mean_ap` on the AP dict and returns a encapsulating dict:
        {'class': {<class_0>: <class_0_AP>, ...}, 'mean': <mean_AP>}
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

    if mean:
        return mean_ap(average_precisions_by_class)
    return average_precisions_by_class


@populationwise
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
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        class_list=class_list,
        mean=False,
    )
    return np.fromiter(ap_per_class.values(), dtype=float).mean()


def armory_to_tide(y_dict, image_id, is_detection=True):
    """
    Convert y_dict in Armory format to use as input for TIDE data type

    y_dict is a dictionary of lists, which needs to be converted to a list of dictionaries
    with keys that correspond to arguments for pushing an element to a TIDE data type
    """

    y_tidecv_list = []

    # convert dictionary with values of list type to list of dictionaries
    # y_dict = {'area': [936, 385]
    #           'boxes': [array([917., 464., 955., 527.], dtype=float32),
    #                     array([911., 468., 940., 517.], dtype=float32)],
    #           'id': [97, 98]
    #           'image_id': [84190894, 84190894],
    #           'is_crowd': [False, True],
    #           'labels': [1, 1]}
    # t = (936, array([917., 464., 955., 527.], dtype=float32), 97, 84190894, False, 1)
    #     (385, array([911., 468., 940., 517.], dtype=float32), 98, 84190894, True, 1)
    # y = {'area': 936, 'boxes': array([917., 464., 955., 527.], dtype=float32), 'id': 97, 'image_id': 84190894, 'is_crowd': False, 'labels': 1}
    #     {'area': 385, 'boxes': array([911., 468., 940., 517.], dtype=float32), 'id': 98, 'image_id': 84190894, 'is_crowd': True, 'labels': 1}
    for y in [dict(zip(y_dict, t)) for t in zip(*y_dict.values())]:
        x1, y1, x2, y2 = y["boxes"]
        width = abs(x1 - x2)
        height = abs(y1 - y2)
        x_min = min(x1, x2)
        y_min = min(y1, y2)

        y_tidecv = {
            "image_id": image_id,
            "class_id": y["labels"],
            "box": [x_min, y_min, width, height],
        }

        if is_detection:
            y_tidecv["score"] = y["scores"]

        y_tidecv_list.append(y_tidecv)
    return y_tidecv_list


@populationwise
def object_detection_mAP_tide(y_list, y_pred_list):
    """
    TIDE version of mean average precision for object detection [https://dbolya.github.io/tide/].

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.

    returns: a dictionary with mean average precision (mAP) for a range of IOU thresholds in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
             and error metrics that weight the significance of different types of errors to assess trade-offs between design choices for models
             as well as potentially for attacks and defense

    default max_dets=100 in TIDE
    max_dets is an attribute for both data_ground_truth and data_detection
    data_ground_truth.max_dets gets passed to tide.evaluate_range > TIDERun._run > TIDERun._eval_image > TIDEEXample._run (code snippet below)
        preds = preds[:max_dets]
        self.preds = preds # Update internally so TIDERun can update itself if :max_dets takes effect
    for now assume we don't want max_dets to affect our metrics
    set data_ground_truth.max_dets and data_detection.max_dets to max of max(len(y["labels"]), len(y_pred["labels"])))
    after going through for loop
    """

    data_ground_truth_name = "ground_truth"
    data_detection_name = "detection"

    data_ground_truth = tidecv.data.Data(name=data_ground_truth_name)
    data_detection = tidecv.data.Data(name=data_detection_name)

    max_dets = 0
    for i, (y, y_pred) in enumerate(zip(y_list, y_pred_list)):
        if "image_id" in y:
            image_id = y["image_id"][0]
        else:
            image_id = i

        tidecv_ground_truth_list = armory_to_tide(y, image_id, is_detection=False)
        for y_tidecv in tidecv_ground_truth_list:
            data_ground_truth.add_ground_truth(**y_tidecv)

        tidecv_detection_list = armory_to_tide(y_pred, image_id)
        for y_tidecv in tidecv_detection_list:
            data_detection.add_detection(**y_tidecv)

        max_len = max(len(y["labels"]), len(y_pred["labels"]))
        if max_len > max_dets:
            max_dets = max_len

    data_ground_truth.max_dets = max_dets
    data_detection.max_dets = max_dets

    tide = TIDE()
    tide.evaluate_range(data_ground_truth, data_detection, mode=TIDE.BOX)
    tide_error = tide.get_all_errors()
    tide_error_count = {
        k.short_name: len(v)
        for k, v in tide.runs[data_detection.name].error_dict.items()
    }
    tide_fn_count = sum(
        [len(v) for k, v in tide.runs[data_detection.name].false_negatives.items()]
    )

    armory_output = {
        "mAP": {
            x.pos_thresh: np.around(x.ap / 100, decimals=2)
            for x in tide.run_thresholds[data_detection.name]
        },
        # not changing numeric format for error yet until we understand what the numbers mean
        "errors": {
            "main": {
                "dAP": {
                    k: np.around(v / 100, decimals=4)
                    for k, v in tide_error["main"][data_detection.name].items()
                },
                "count": tide_error_count,
            },
            "special": {
                "dAP": {
                    k: np.around(v / 100, decimals=4)
                    for k, v in tide_error["special"][data_detection.name].items()
                },
                "count": {"FalseNeg": tide_fn_count},
            },
        },
    }

    return armory_output


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


@populationwise
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


@populationwise
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


@populationwise
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


@populationwise
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


@populationwise
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


@populationwise
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


@populationwise
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


@populationwise
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


@populationwise
def carla_od_AP_per_class(y_list, y_pred_list, iou_threshold=0.5, mean=True):
    class_list = [1, 2, 3]
    """
    CARLA object detection datasets contains class labels 1-4, with class 4 representing
    the green screen/patch itself, which should not be treated as an object class.
    """
    return object_detection_AP_per_class(
        y_list,
        y_pred_list,
        iou_threshold=iou_threshold,
        class_list=class_list,
        mean=mean,
    )


@populationwise
def apricot_patch_targeted_AP_per_class(
    y_list, y_pred_list, iou_threshold=0.1, mean=True
):
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

    mean: if False, returns a dict mapping each class to its average precision (AP)
        if True, calls `mean_ap` on the AP dict and returns a encapsulating dict:
        {'class': {<class_0>: <class_0_AP>, ...}, 'mean': <mean_AP>}
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

    if mean:
        return mean_ap(average_precisions_by_class)
    return average_precisions_by_class


@populationwise
def dapricot_patch_targeted_AP_per_class(
    y_list, y_pred_list, iou_threshold=0.1, mean=True
):
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

    mean: if False, returns a dict mapping each class to its average precision (AP)
        if True, calls `mean_ap` on the AP dict and returns a encapsulating dict:
        {'class': {<class_0>: <class_0_AP>, ...}, 'mean': <mean_AP>}
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

    if mean:
        return mean_ap(average_precisions_by_class)
    return average_precisions_by_class


@populationwise
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


class HOTA_metrics:
    def __init__(self, tracked_classes=("pedestrian",), coco_format: bool = False):
        from collections import defaultdict

        from TrackEval.trackeval.datasets._base_dataset import _BaseDataset

        # TrackEval repo: https://github.com/JonathonLuiten/TrackEval
        from TrackEval.trackeval.metrics.hota import HOTA

        from armory.data.adversarial_datasets import mot_coco_to_array

        self.class_name_to_class_id = {"pedestrian": 1, "vehicle": 2}
        self.tracked_classes = list(tracked_classes)
        self.coco_format = bool(coco_format)
        self.hota_metrics_per_class_per_videos = {
            key: defaultdict(dict) for key in self.tracked_classes
        }
        self.hota_metrics_per_class_all_videos = {
            key: {} for key in self.tracked_classes
        }
        self.HOTA_calc = HOTA()
        self._BaseDataset = _BaseDataset
        self.mot_coco_to_array = mot_coco_to_array

    def _check_and_format(self, data):
        if self.coco_format:
            data = self.mot_coco_to_array(data)

        if data.ndim == 3:
            if len(data) != 1:
                raise ValueError("Batch size > 1 not currently supported")
            data = data[0]
        if data.ndim != 2:
            raise ValueError(f"Input data must be 2D or 3D, not {data.ndim}")
        return data

    @staticmethod
    def relabel(unique_ids, list_of_id_arrays):
        """
        Re-label int IDs to 0-indexed sequential IDs
        """
        unique_ids = sorted(set(unique_ids))
        new_unique_ids = list(range(len(unique_ids)))
        id_map = {k: v for k, v in zip(unique_ids, new_unique_ids)}
        new_list = [
            np.array([id_map[x] for x in array], dtype=array.dtype)
            for array in list_of_id_arrays
        ]
        return new_unique_ids, new_list

    def preprocess(self, gt_data, tracker_data, tracked_class):
        """
        This function preprocesses data into a format required for HOTA metrics calculation.

        It simplifies get_preprocessed_seq_data() at https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/mot_challenge_2d_box.py
        and tailors it to MOT data generated using CARLA.

        After preprocessing, the HOTA metrics are caluclated - ref: https://link.springer.com/article/10.1007/s11263-020-01375-2

        Inputs:
            - gt_data and tracker_data are 2D NDArrays, where each row is a detection in the format of:
                <timestep> <object_id> <bbox top-left x> <bbox top-left y> <bbox width> <bbox height> <confidence_score=1> <class_id> <visibility=1>
            - tracked_class is a string representing the class for which HOTA is calculated.
        """
        gt_data = self._check_and_format(gt_data)
        tracker_data = self._check_and_format(tracker_data)

        cls_id = self.class_name_to_class_id[tracked_class]

        # NOTE: assumes there are ground truth predictions in every frame
        num_timesteps = len(set(gt_data[:, 0]))
        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "tracker_confidences",
            "similarity_scores",
        ]
        data = {key: [None] * num_timesteps for key in data_keys}
        unique_gt_ids, unique_tracker_ids = set(), set()
        num_gt_dets, num_tracker_dets = 0, 0
        for t in range(num_timesteps):
            # Get all data
            t_index = gt_data[:, 0] == t
            gt_ids, gt_dets, gt_classes = (
                gt_data[t_index, x] for x in (1, slice(2, 6), 7)
            )

            t_index = tracker_data[:, 0] == t
            tracker_ids, tracker_dets, tracker_confidences, tracker_classes = (
                tracker_data[t_index, x] for x in (1, slice(2, 6), 6, 7)
            )

            similarity_scores = self._BaseDataset._calculate_box_ious(
                gt_dets, tracker_dets, box_format="xywh"
            )

            # Keep only tracker associated with given class
            tracker_to_keep_mask = np.equal(tracker_classes, cls_id)
            data["tracker_ids"][t] = tracker_ids[tracker_to_keep_mask].astype(int)
            data["tracker_dets"][t] = tracker_dets[tracker_to_keep_mask]
            data["tracker_confidences"][t] = tracker_confidences[tracker_to_keep_mask]

            # Keep only detections associated with given class
            gt_to_keep_mask = np.equal(gt_classes, cls_id)
            data["gt_ids"][t] = gt_ids[gt_to_keep_mask].astype(int)
            data["gt_dets"][t] = gt_dets[gt_to_keep_mask, :]
            data["similarity_scores"][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids.update(data["gt_ids"][t])
            unique_tracker_ids.update(data["tracker_ids"][t])
            num_tracker_dets += len(data["tracker_ids"][t])
            num_gt_dets += len(data["gt_ids"][t])

        # Re-label int IDs to 1-indexed sequential IDs
        unique_gt_ids, data["gt_ids"] = self.relabel(unique_gt_ids, data["gt_ids"])
        unique_tracker_ids, data["tracker_ids"] = self.relabel(
            unique_tracker_ids, data["tracker_ids"]
        )

        # Record overview statistics.
        data.update(
            {
                "num_tracker_dets": num_tracker_dets,
                "num_gt_dets": num_gt_dets,
                "num_tracker_ids": len(unique_tracker_ids),
                "num_gt_ids": len(unique_gt_ids),
                "num_timesteps": num_timesteps,
            }
        )

        # Ensure ids are unique per timestep after preproc.
        self._BaseDataset._check_unique_ids(data, after_preproc=True)

        return data

    # Function to calculate the main HOTA metric and its component sub-metrics
    def calculate_hota_metrics_per_class_per_video(
        self, gt_data, tracker_data, tracked_class, video_name
    ):

        # Calculate per-video HOTA metrics
        data = self.preprocess(gt_data, tracker_data, tracked_class)
        self.hota_metrics_per_class_per_videos[tracked_class][
            video_name
        ] = self.HOTA_calc.eval_sequence(data)

    def calculate_hota_metrics_per_class_all_videos(self, tracked_class):
        self.hota_metrics_per_class_all_videos[
            tracked_class
        ] = self.HOTA_calc.combine_sequences(
            self.hota_metrics_per_class_per_videos[tracked_class]
        )

    def get_per_class_per_video_metrics(self):
        return self.hota_metrics_per_class_per_videos

    def get_per_class_all_videos_metrics(self):
        return self.hota_metrics_per_class_all_videos


class GlobalHOTA:
    # there are many HOTA sub-metrics. We care mostly about the mean values of these three.
    METRICS = ("hota", "deta", "assa")

    def __init__(
        self,
        metrics=("hota", "deta", "assa"),
        means=True,
        record_metric_per_sample=True,
        **kwargs,
    ):
        for k in metrics:
            if k not in self.METRICS:
                raise ValueError(f"{k} not in {self.METRICS}")
        self.metrics = tuple(metrics)
        self.means = bool(means)
        self.record_metric_per_sample = bool(record_metric_per_sample)

        self.hota_metrics = HOTA_metrics(**kwargs)

    def __call__(self, y_list, y_pred_list):
        for i, (y, y_pred) in enumerate(zip(y_list, y_pred_list)):
            for tracked_class in self.hota_metrics.tracked_classes:
                self.hota_metrics.calculate_hota_metrics_per_class_per_video(
                    y, y_pred, tracked_class, i
                )

        for tracked_class in self.hota_metrics.tracked_classes:
            self.hota_metrics.calculate_hota_metrics_per_class_all_videos(tracked_class)

        results = {}
        if self.record_metric_per_sample:
            per_class_per_video_metrics = (
                self.hota_metrics.get_per_class_per_video_metrics()
            )
            for tracked_class in self.hota_metrics.tracked_classes:
                for k in ["hota", "deta", "assa"]:
                    results[f"{k}"] = []
                for vid in per_class_per_video_metrics[tracked_class].keys():
                    for k in ["HOTA", "DetA", "AssA"]:
                        value = per_class_per_video_metrics[tracked_class][vid][
                            k
                        ].mean()
                        results[f"{k.lower()}"].append(value)

        if self.means:
            per_class_all_videos_metrics = (
                self.hota_metrics.get_per_class_all_videos_metrics()
            )
            for tracked_class in self.hota_metrics.tracked_classes:
                for k in ["HOTA", "DetA", "AssA"]:
                    value = per_class_all_videos_metrics[tracked_class][k].mean()
                    results[f"mean_{k.lower()}"] = value

        return results


populationwise(GlobalHOTA, name="hota_metrics")


@result_formatter("total_wer")
def total_wer_formatter(result):
    total, (num, denom) = result
    return f"total={float(total):.3}, {num}/{denom}"


@result_formatter("word_error_rate")
def word_error_rate_formatter(result):
    result = total_wer(result)
    return total_wer_formatter(result)


@result_formatter("hota_metrics")
def hota_metrics_formatter(result):
    mean_results = {k: v for k, v in result.items() if "mean" in k}
    return f"{mean_results}"


@result_formatter("total_entailment")
def total_entailment_formatter(result):
    total = sum(result.values())
    return (
        f"contradiction: {result['contradiction']}/{total}, "
        f"neutral: {result['neutral']}/{total}, "
        f"entailment: {result['entailment']}/{total}"
    )


@result_formatter("entailment")
def entailment_formatter(result):
    result = total_entailment(result)
    return total_entailment_formatter(result)


map_to_aggregator("entailment", "total_entailment")
map_to_aggregator("word_error_rate", "total_wer")
