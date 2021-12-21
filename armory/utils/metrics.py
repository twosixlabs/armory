"""
Metrics for scenarios

Outputs are lists of python variables amenable to JSON serialization:
    e.g., bool, int, float
    numpy data types and tensors generally fail to serialize
"""

import logging
import numpy as np
import time
from contextlib import contextmanager
import io
from collections import defaultdict, Counter

import cProfile
import pstats

from armory.data.adversarial_datasets import ADV_PATCH_MAGIC_NUMBER_LABEL_ID
from armory.data.adversarial.apricot_metadata import APRICOT_PATCHES


logger = logging.getLogger(__name__)


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


def norm(x, x_adv, ord):
    """
    Return the given norm over a batch, outputting a list of floats
    """
    x = np.asarray(x)
    x_adv = np.asarray(x_adv)
    # elevate to 64-bit types first to prevent overflow errors
    assert not (
        np.iscomplexobj(x) ^ np.iscomplexobj(x_adv)
    ), "x and x_adv mix real/complex types"
    dtype = complex if np.iscomplexobj(x) else float
    diff = (x.astype(dtype) - x_adv.astype(dtype)).reshape(x.shape[0], -1)
    values = np.linalg.norm(diff, ord=ord, axis=1)
    # normalize l0 norm by number of elements in array
    if ord == 0:
        return list(float(x) / diff[i].size for i, x in enumerate(values))
    return list(float(x) for x in values)


def linf(x, x_adv):
    """
    Return the L-infinity norm over a batch of inputs as a float
    """
    return norm(x, x_adv, np.inf)


def l2(x, x_adv):
    """
    Return the L2 norm over a batch of inputs as a float
    """
    return norm(x, x_adv, 2)


def l1(x, x_adv):
    """
    Return the L1 norm over a batch of inputs as a float
    """
    return norm(x, x_adv, 1)


def lp(x, x_adv, p):
    """
    Return the Lp norm over a batch of inputs as a float
    """
    if p <= 0:
        raise ValueError(f"p must be positive, not {p}")
    return norm(x, x_adv, p)


def l0(x, x_adv):
    """
    Return the L0 'norm' over a batch of inputs as a float,
    normalized by the number of elements in the array
    """
    return norm(x, x_adv, 0)


def _snr(x_i, x_adv_i):
    assert not (
        np.iscomplexobj(x_i) ^ np.iscomplexobj(x_adv_i)
    ), "x_i and x_adv_i mix real/complex types"
    dtype = complex if np.iscomplexobj(x_i) else float
    x_i = np.asarray(x_i, dtype=dtype)
    x_adv_i = np.asarray(x_adv_i, dtype=dtype)
    if x_i.shape != x_adv_i.shape:
        raise ValueError(f"x_i.shape {x_i.shape} != x_adv_i.shape {x_adv_i.shape}")
    signal_power = (np.abs(x_i) ** 2).mean()
    noise_power = (np.abs(x_i - x_adv_i) ** 2).mean()
    if noise_power == 0:
        return np.inf
    return signal_power / noise_power


def snr(x, x_adv):
    """
    Return the SNR of a batch of samples with raw audio input
    """
    if len(x) != len(x_adv):
        raise ValueError(f"len(x) {len(x)} != len(x_adv) {len(x_adv)}")
    return [float(_snr(x_i, x_adv_i)) for (x_i, x_adv_i) in zip(x, x_adv)]


def snr_db(x, x_adv):
    """
    Return the SNR of a batch of samples with raw audio input in Decibels (DB)
    """
    return [float(i) for i in 10 * np.log10(snr(x, x_adv))]


def _snr_spectrogram(x_i, x_adv_i):
    x_i = np.asarray(x_i, dtype=float)
    x_adv_i = np.asarray(x_adv_i, dtype=float)
    if x_i.shape != x_adv_i.shape:
        raise ValueError(f"x_i.shape {x_i.shape} != x_adv_i.shape {x_adv_i.shape}")
    signal_power = np.abs(x_i).mean()
    noise_power = np.abs(x_i - x_adv_i).mean()
    return signal_power / noise_power


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


# Metrics specific to MARS model preprocessing in video UCF101 scenario


def verify_mars(x, x_adv):
    if len(x) != len(x_adv):
        raise ValueError(f"len(x) {len(x)} != {len(x_adv)} len(x_adv)")
    for x_i, x_adv_i in zip(x, x_adv):
        if x_i.shape[1:] != x_adv_i.shape[1:]:
            raise ValueError(f"Shape {x_i.shape[1:]} != {x_adv_i.shape[1:]}")
        if x_i.shape[1:] != (3, 16, 112, 112):
            raise ValueError(f"Shape {x_i.shape[1:]} != (3, 16, 112, 112)")


def mars_mean_l2(x, x_adv):
    """
    Input dimensions: (n_batch, n_stacks, channels, stack_frames, height, width)
        Typically: (1, variable, 3, 16, 112, 112)
    """
    verify_mars(x, x_adv)
    out = []
    for x_i, x_adv_i in zip(x, x_adv):
        out.append(np.mean(l2(x_i, x_adv_i)))
    return out


def mars_reshape(x_i):
    """
    Reshape (n_stacks, 3, 16, 112, 112) into (n_stacks * 16, 112, 112, 3)
    """
    return np.transpose(x_i, (0, 2, 3, 4, 1)).reshape((-1, 112, 112, 3))


def mars_mean_patch(x, x_adv):
    verify_mars(x, x_adv)
    out = []
    for x_i, x_adv_i in zip(x, x_adv):
        out.append(
            np.mean(
                image_circle_patch_diameter(mars_reshape(x_i), mars_reshape(x_adv_i))
            )
        )
    return out


@contextmanager
def resource_context(name="Name", profiler=None, computational_resource_dict=None):
    if profiler is None:
        yield
        return 0
    profiler_types = ["Basic", "Deterministic"]
    if profiler is not None and profiler not in profiler_types:
        raise ValueError(f"Profiler {profiler} is not one of {profiler_types}.")
    if profiler == "Deterministic":
        logger.warn(
            "Using Deterministic profiler. This may reduce timing accuracy and result in a large results file."
        )
        pr = cProfile.Profile()
        pr.enable()
    startTime = time.perf_counter()
    yield
    elapsedTime = time.perf_counter() - startTime
    if profiler == "Deterministic":
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        stats = s.getvalue()
    if name not in computational_resource_dict:
        computational_resource_dict[name] = defaultdict(lambda: 0)
        if profiler == "Deterministic":
            computational_resource_dict[name]["stats"] = ""
    comp = computational_resource_dict[name]
    comp["execution_count"] += 1
    comp["total_time"] += elapsedTime
    if profiler == "Deterministic":
        comp["stats"] += stats
    return 0


def snr_spectrogram(x, x_adv):
    """
    Return the SNR of a batch of samples with spectrogram input

    NOTE: Due to phase effects, this is only an estimate of the SNR.
        For instance, if x[0] = sin(t) and x_adv[0] = sin(t + 2*pi/3),
        Then the SNR will be calculated as infinity, when it should be 1.
        However, the spectrograms will look identical, so as long as the
        model uses spectrograms and not the underlying raw signal,
        this should not have a significant effect on the results.
    """
    if x.shape != x_adv.shape:
        raise ValueError(f"x.shape {x.shape} != x_adv.shape {x_adv.shape}")
    return [float(_snr_spectrogram(x_i, x_adv_i)) for (x_i, x_adv_i) in zip(x, x_adv)]


def snr_spectrogram_db(x, x_adv):
    """
    Return the SNR of a batch of samples with spectrogram input in Decibels (DB)
    """
    return [float(i) for i in 10 * np.log10(snr_spectrogram(x, x_adv))]


def _image_circle_patch_diameter(x_i, x_adv_i):
    if x_i.shape != x_adv_i.shape:
        raise ValueError(f"x_i.shape {x_i.shape} != x_adv_i.shape {x_adv_i.shape}")
    img_shape = x_i.shape
    if len(img_shape) != 3:
        raise ValueError(f"Expected image with 3 dimensions. x_i has shape {x_i.shape}")
    if (x_i == x_adv_i).mean() < 0.5:
        logger.warning(
            f"x_i and x_adv_i differ at {int(100*(x_i != x_adv_i).mean())} percent of "
            "indices. image_circle_patch_area may not be accurate"
        )
    # Identify which axes of input array are spatial vs. depth dimensions
    depth_dim = img_shape.index(min(img_shape))
    spat_ind = 1 if depth_dim != 1 else 0

    # Determine which indices (along the spatial dimension) are perturbed
    pert_spatial_indices = set(np.where(x_i != x_adv_i)[spat_ind])
    if len(pert_spatial_indices) == 0:
        logger.warning("x_i == x_adv_i. image_circle_patch_area is 0")
        return 0

    # Find which indices (preceding the patch's max index) are unperturbed, in order
    # to determine the index of the edge of the patch
    max_ind_of_patch = max(pert_spatial_indices)
    unpert_ind_less_than_patch_max_ind = [
        i for i in range(max_ind_of_patch) if i not in pert_spatial_indices
    ]
    min_ind_of_patch = (
        max(unpert_ind_less_than_patch_max_ind) + 1
        if unpert_ind_less_than_patch_max_ind
        else 0
    )

    # If there are any perturbed indices outside the range of the patch just computed
    if min(pert_spatial_indices) < min_ind_of_patch:
        logger.warning("Multiple regions of the image have been perturbed")

    diameter = max_ind_of_patch - min_ind_of_patch + 1
    spatial_dims = [dim for i, dim in enumerate(img_shape) if i != depth_dim]
    patch_diameter = diameter / min(spatial_dims)
    return patch_diameter


def image_circle_patch_diameter(x, x_adv):
    """
    Returns diameter of circular image patch, normalized by the smaller spatial dimension
    """
    return [
        _image_circle_patch_diameter(x_i, x_adv_i) for (x_i, x_adv_i) in zip(x, x_adv)
    ]


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
    "dapricot_patch_target_success": dapricot_patch_target_success,
    "dapricot_patch_targeted_AP_per_class": dapricot_patch_targeted_AP_per_class,
    "apricot_patch_targeted_AP_per_class": apricot_patch_targeted_AP_per_class,
    "categorical_accuracy": categorical_accuracy,
    "top_n_categorical_accuracy": top_n_categorical_accuracy,
    "top_5_categorical_accuracy": top_5_categorical_accuracy,
    "norm": norm,
    "l0": l0,
    "l1": l1,
    "l2": l2,
    "lp": lp,
    "linf": linf,
    "video_tracking_mean_iou": video_tracking_mean_iou,
    "snr": snr,
    "snr_db": snr_db,
    "snr_spectrogram": snr_spectrogram,
    "snr_spectrogram_db": snr_spectrogram_db,
    "image_circle_patch_diameter": image_circle_patch_diameter,
    "mars_mean_l2": mars_mean_l2,
    "mars_mean_patch": mars_mean_patch,
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
}

# Image-based metrics applied to video


def video_metric(metric, frame_average="mean"):
    mapping = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
    }
    if frame_average not in mapping:
        raise ValueError(f"frame_average {frame_average} not in {tuple(mapping)}")
    frame_average_func = mapping[frame_average]

    def func(x, x_adv):
        results = []
        for x_sample, x_adv_sample in zip(x, x_adv):
            frames = metric(x_sample, x_adv_sample)
            results.append(frame_average_func(frames))
        return results

    return func


for metric_name in "l0", "l1", "l2", "linf", "image_circle_patch_diameter":
    metric = SUPPORTED_METRICS[metric_name]
    for prefix in "mean", "max":
        new_metric_name = prefix + "_" + metric_name
        if new_metric_name in SUPPORTED_METRICS:
            raise ValueError(f"Duplicate metric {new_metric_name} in SUPPORTED_METRICS")
        new_metric = video_metric(metric, frame_average=prefix)
        SUPPORTED_METRICS[new_metric_name] = new_metric


class MetricList:
    """
    Keeps track of all results from a single metric
    """

    def __init__(self, name, function=None):
        if function is None:
            try:
                self.function = SUPPORTED_METRICS[name]
            except KeyError:
                raise KeyError(f"{name} is not part of armory.utils.metrics")
        elif callable(function):
            self.function = function
        else:
            raise ValueError(f"function must be callable or None, not {function}")
        self.name = name
        self._values = []
        self._input_labels = []
        self._input_preds = []

    def clear(self):
        self._values.clear()

    def add_results(self, *args, **kwargs):
        value = self.function(*args, **kwargs)
        self._values.extend(value)

    def __iter__(self):
        return self._values.__iter__()

    def __len__(self):
        return len(self._values)

    def values(self):
        return list(self._values)

    def mean(self):
        if not self._values:
            return float("nan")
        return sum(float(x) for x in self._values) / len(self._values)

    def append_input_label(self, label):
        self._input_labels.extend(label)

    def append_input_pred(self, pred):
        self._input_preds.extend(pred)

    def total_wer(self):
        # checks if all values are tuples from the WER metric
        if all(isinstance(wer_tuple, tuple) for wer_tuple in self._values):
            total_edit_distance = 0
            total_words = 0
            for wer_tuple in self._values:
                total_edit_distance += wer_tuple[0]
                total_words += wer_tuple[1]
            return float(total_edit_distance / total_words)
        else:
            raise ValueError("total_wer() only for WER metric")

    def compute_non_elementwise_metric(self, **kwargs):
        return self.function(self._input_labels, self._input_preds, **kwargs)


class MetricsLogger:
    """
    Uses the set of task and perturbation metrics given to it.
    """

    def __init__(
        self,
        task=None,
        perturbation=None,
        means=True,
        record_metric_per_sample=False,
        profiler_type=None,
        computational_resource_dict=None,
        skip_benign=None,
        skip_attack=None,
        targeted=False,
        task_kwargs=None,
        **kwargs,
    ):
        """
        task - single metric or list of metrics
        perturbation - single metric or list of metrics
        means - whether to return the mean value for each metric
        record_metric_per_sample - whether to return metric values for each sample
        """
        self.tasks = [] if skip_benign else self._generate_counters(task)
        self.adversarial_tasks = [] if skip_attack else self._generate_counters(task)
        self.targeted_tasks = (
            self._generate_counters(task) if targeted and not skip_attack else []
        )
        self.perturbations = (
            [] if skip_attack else self._generate_counters(perturbation)
        )
        self.means = bool(means)
        self.full = bool(record_metric_per_sample)
        self.computational_resource_dict = {}
        if not self.means and not self.full:
            logger.warning(
                "No per-sample metric results will be produced. "
                "To change this, set 'means' or 'record_metric_per_sample' to True."
            )
        if (
            not self.tasks
            and not self.perturbations
            and not self.adversarial_tasks
            and not self.targeted_tasks
        ):
            logger.warning(
                "No metric results will be produced. "
                "To change this, set one or more 'task' or 'perturbation' metrics"
            )
        # the following metrics must be computed at once after all predictions have been obtained
        self.non_elementwise_metrics = [
            "object_detection_AP_per_class",
            "apricot_patch_targeted_AP_per_class",
            "dapricot_patch_targeted_AP_per_class",
        ]
        self.mean_ap_metrics = [
            "object_detection_AP_per_class",
            "apricot_patch_targeted_AP_per_class",
            "dapricot_patch_targeted_AP_per_class",
        ]

        self.task_kwargs = task_kwargs
        if task_kwargs:
            if not isinstance(task_kwargs, list):
                raise TypeError(
                    f"task_kwargs should be of type list, found {type(task_kwargs)}"
                )
            if len(task_kwargs) != len(task):
                raise ValueError(
                    f"task is of length {len(task)} but task_kwargs is of length {len(task_kwargs)}"
                )

        # the following metrics must be computed at once after all predictions have been obtained
        self.non_elementwise_metrics = [
            "object_detection_AP_per_class",
            "apricot_patch_targeted_AP_per_class",
            "dapricot_patch_targeted_AP_per_class",
            "carla_od_AP_per_class",
        ]
        self.mean_ap_metrics = [
            "object_detection_AP_per_class",
            "apricot_patch_targeted_AP_per_class",
            "dapricot_patch_targeted_AP_per_class",
            "carla_od_AP_per_class",
        ]

        # This designation only affects logging formatting
        self.quantity_metrics = [
            "object_detection_hallucinations_per_image",
            "carla_od_hallucinations_per_image",
        ]

        self.task_kwargs = task_kwargs
        if task_kwargs:
            if not isinstance(task_kwargs, list):
                raise TypeError(
                    f"task_kwargs should be of type list, found {type(task_kwargs)}"
                )
            if len(task_kwargs) != len(task):
                raise ValueError(
                    f"task is of length {len(task)} but task_kwargs is of length {len(task_kwargs)}"
                )

        # the following metrics must be computed at once after all predictions have been obtained
        self.non_elementwise_metrics = [
            "object_detection_AP_per_class",
            "apricot_patch_targeted_AP_per_class",
            "dapricot_patch_targeted_AP_per_class",
            "carla_od_AP_per_class",
        ]
        self.mean_ap_metrics = [
            "object_detection_AP_per_class",
            "apricot_patch_targeted_AP_per_class",
            "dapricot_patch_targeted_AP_per_class",
            "carla_od_AP_per_class",
        ]

        # This designation only affects logging formatting
        self.quantity_metrics = [
            "object_detection_hallucinations_per_image",
            "carla_od_hallucinations_per_image",
        ]

    def _generate_counters(self, names):
        if names is None:
            names = []
        elif isinstance(names, str):
            names = [names]
        elif not isinstance(names, list):
            raise ValueError(
                f"{names} must be one of (None, str, list), not {type(names)}"
            )
        return [MetricList(x) for x in names]

    @classmethod
    def from_config(cls, config, skip_benign=None, skip_attack=None, targeted=None):
        if skip_benign:
            config["skip_benign"] = skip_benign
        if skip_attack:
            config["skip_attack"] = skip_attack
        return cls(**config, targeted=targeted)

    def clear(self):
        for metric in self.tasks + self.adversarial_tasks + self.perturbations:
            metric.clear()

    def update_task(self, y, y_pred, adversarial=False, targeted=False):
        if targeted and not adversarial:
            raise ValueError("benign task cannot be targeted")
        tasks = (
            self.targeted_tasks
            if targeted
            else self.adversarial_tasks
            if adversarial
            else self.tasks
        )
        for task_idx, metric in enumerate(tasks):
            if metric.name in self.non_elementwise_metrics:
                metric.append_input_label(y)
                metric.append_input_pred(y_pred)
            else:
                if self.task_kwargs:
                    metric.add_results(y, y_pred, **self.task_kwargs[task_idx])
                else:
                    metric.add_results(y, y_pred)

    def update_perturbation(self, x, x_adv):
        for metric in self.perturbations:
            metric.add_results(x, x_adv)

    def log_task(self, adversarial=False, targeted=False, used_preds_as_labels=False):
        if used_preds_as_labels and not adversarial:
            raise ValueError("benign task shouldn't use benign predictions as labels")
        if used_preds_as_labels and targeted:
            raise ValueError("targeted task shouldn't use benign predictions as labels")
        if targeted:
            if adversarial:
                metrics = self.targeted_tasks
                wrt = "target"
                task_type = "adversarial"
            else:
                raise ValueError("benign task cannot be targeted")
        elif adversarial:
            metrics = self.adversarial_tasks
            if used_preds_as_labels:
                wrt = "benign predictions as"
            else:
                wrt = "ground truth"
            task_type = "adversarial"
        else:
            metrics = self.tasks
            wrt = "ground truth"
            task_type = "benign"

        for task_idx, metric in enumerate(metrics):
            # Do not calculate mean WER, calcuate total WER
            if metric.name == "word_error_rate":
                logger.info(
                    f"Word error rate on {task_type} examples relative to {wrt} labels: "
                    f"{metric.total_wer():.2%}"
                )
            elif metric.name in self.non_elementwise_metrics:
                if self.task_kwargs:
                    metric_result = metric.compute_non_elementwise_metric(
                        **self.task_kwargs[task_idx]
                    )
                else:
                    metric_result = metric.compute_non_elementwise_metric()
                logger.info(
                    f"{metric.name} on {task_type} test examples relative to {wrt} labels: "
                    f"{metric_result}"
                )
                if metric.name in self.mean_ap_metrics:
                    logger.info(
                        f"mean {metric.name} on {task_type} examples relative to {wrt} labels "
                        f"{np.fromiter(metric_result.values(), dtype=float).mean():.2%}."
                    )
            elif metric.name in self.quantity_metrics:
                # Don't include % symbol
                logger.info(
                    f"Average {metric.name} on {task_type} test examples relative to {wrt} labels: "
                    f"{metric.mean():.2}"
                )
                if metric.name in self.mean_ap_metrics:
                    logger.info(
                        f"mean {metric.name} on {task_type} examples relative to {wrt} labels "
                        f"{np.fromiter(metric_result.values(), dtype=float).mean():.2%}."
                    )
            else:
                logger.info(
                    f"Average {metric.name} on {task_type} test examples relative to {wrt} labels: "
                    f"{metric.mean():.2%}"
                )

    def results(self):
        """
        Return dict of results
        """
        results = {}
        for metrics, prefix in [
            (self.tasks, "benign"),
            (self.adversarial_tasks, "adversarial"),
            (self.targeted_tasks, "targeted"),
            (self.perturbations, "perturbation"),
        ]:
            for task_idx, metric in enumerate(metrics):
                if metric.name in self.non_elementwise_metrics:
                    if self.task_kwargs:
                        metric_result = metric.compute_non_elementwise_metric(
                            **self.task_kwargs[task_idx]
                        )
                    else:
                        metric_result = metric.compute_non_elementwise_metric()
                    results[f"{prefix}_{metric.name}"] = metric_result
                    if metric.name in self.mean_ap_metrics:
                        results[f"{prefix}_mean_{metric.name}"] = np.fromiter(
                            metric_result.values(), dtype=float
                        ).mean()
                    continue

                if self.full:
                    results[f"{prefix}_{metric.name}"] = metric.values()
                if self.means:
                    try:
                        results[f"{prefix}_mean_{metric.name}"] = metric.mean()
                    except ZeroDivisionError:
                        raise ZeroDivisionError(
                            f"No values to calculate mean in {prefix}_{metric.name}"
                        )
                if metric.name == "word_error_rate":
                    try:
                        results[f"{prefix}_total_{metric.name}"] = metric.total_wer()
                    except ZeroDivisionError:
                        raise ZeroDivisionError(
                            f"No values to calculate WER in {prefix}_{metric.name}"
                        )

        for name in self.computational_resource_dict:
            entry = self.computational_resource_dict[name]
            if "execution_count" not in entry or "total_time" not in entry:
                raise ValueError(
                    "Computational resource dictionary entry corrupted, missing data."
                )
            total_time = entry["total_time"]
            execution_count = entry["execution_count"]
            average_time = total_time / execution_count
            results[
                f"Avg. CPU time (s) for {execution_count} executions of {name}"
            ] = average_time
            if "stats" in entry:
                results[f"{name} profiler stats"] = entry["stats"]
        return results
