"""
Task metrics
"""

import collections
import functools
import logging

import numpy as np

from armory.metrics.perturbation import MetricNameSpace, as_batch


logger = logging.getLogger(__name__)


element = MetricNameSpace()
batch = MetricNameSpace()
aggregate = MetricNameSpace()


def aggregator(aggregate_function, name=None):
    """
    Register an aggregate function and register it
    """
    if name is None:
        name = aggregate_function.__name__
    setattr(aggregate, name, aggregate_function)


def batchwise(batch_metric, name=None):
    """
    Register a batch metric and register a batchwise version of it
    """
    if name is None:
        name = batch_metric.__name__
    setattr(batch, name, batch_metric)


def elementwise(element_metric, name=None):
    """
    Register a element metric and register a batchwise version of it
    """
    if name is None:
        name = element_metric.__name__
    setattr(element, name, element_metric)
    batch_metric = as_batch(element_metric)
    batchwise(batch_metric, name=name)
    return element_metric


def numpy(function):
    """
    Ensures args (but not kwargs) are passed in as numpy vectors

    In contrast to perturbation.numpy, this does not ensure same shape or dtype
    """

    @functools.wraps(function)
    def wrapper(y, y_pred, **kwargs):
        y, y_pred = (np.asarray(i) for i in (y, y_pred))
        return function(y, y_pred, **kwargs)

    return wrapper


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
    return y == y_pred


@elementwise
def top_5_categorical_accuracy(y, y_pred):
    """
    Return the top 5 categorical accuracy of the predictions
    """
    return top_n_categorical_accuracy(y, y_pred, n=5)


@numpy
def top_n_categorical_accuracy(y, y_pred, n=5):
    """
    Return the top n categorical accuracy of the predictions

    y_pred - must be a vector of values
    """
    if n < 1 or n == np.inf or n != int(n):
        raise ValueError(f"n must be a positive integer, not {n}")
    n = int(n)

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
    if isinstance(y, str):
        reference = y.split()
    elif isinstance(y, bytes):
        reference = y.decode("utf-8").split()
    else:
        raise TypeError(f"y is of type {type(y)}, expected string or bytes")
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


@aggregator
def total_wer(word_error_rate_tuples):
    total_edit_distance = 0
    total_words = 0
    for wer_tuple in word_error_rate_tuples:
        if not isinstance(wer_tuple, tuple) or len(wer_tuple) != 2:
            raise ValueError(f"{wer_tuple} is not a tuple from word_error_rate()")
        total_edit_distance += wer_tuple[0]
        total_words += wer_tuple[1]
    if total_words == 0:
        return float("nan")
    return float(total_edit_distance / total_words)


@aggregator
def mean(values):
    if len(values) == 0:
        return float("nan")
    return sum(float(x) for x in values) / len(values)


@aggregator
def mean_ap(dictionary):
    if len(dictionary) == 0:
        return float("nan")
    return np.fromiter(dictionary.values(), dtype=float).mean()


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
        logger.warning(
            "One set of boxes appears to be normalized while the other is not"
        )

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


def object_detection_AP_per_class(y_list, y_pred_list, iou_threshold=0.5):
    """
    Mean average precision for object detection. This function returns a dictionary
    mapping each class to the average precision (AP) for the class. The mAP can be computed
    by taking the mean of the AP's across all classes. This metric is computed over all
    evaluation samples, rather than on a per-sample basis.

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    """
    from armory.data.adversarial_datasets import ADV_PATCH_MAGIC_NUMBER_LABEL_ID

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
        num_gt_boxes_per_img = collections.Counter(
            [gt["img_idx"] for gt in class_gt_boxes]
        )

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
        num_patch_boxes_per_img = collections.Counter(
            [gt["img_idx"] for gt in class_patch_boxes]
        )

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
