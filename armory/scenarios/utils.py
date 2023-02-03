from collections import defaultdict
from typing import Tuple

import numpy as np


def to_categorical(y, num_classes=None, dtype=np.float32):
    y = np.asarray(y, dtype="int")
    if y.ndim != 1:
        raise ValueError("y is not a 1D array")
    if not num_classes:
        num_classes = np.max(y) + 1
    return np.eye(num_classes, dtype=dtype)[y]


def from_categorical(y, dtype=int):
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError("y is not a 2D array")
    return np.argmax(y, axis=1)


# Train set splitting utilities

# Given a dataset with only two splits - train and test, these utilities divide the train set
# into subsets following a generalized version of the logic from the Bullseye Polytope
# attack, which in turn borrows the logic from the Convex Polytope attack, available at
# https://arxiv.org/pdf/1905.05897.pdf for CIFAR10. The motivating idea is, to benchmark
# an attack it is important to repeat the experiment with multiple targets (recall
# the target is a single datapoint). For consistency each experiment should train
# on identical images (except for the poisons), and for measurement purposes the benign performance
# on the test set is measured against only the test set. So the poisons (whose exact number
# will vary based on the strength of the attack) should be pulled from a fixed subset of the
# training data, as should the targets. The paper also describes using a subset of the dataset
# to fine-tune on - since this is not a realistic use case of fine-tuning, the utility methods
# do not implement this functionality.


def split_train_target(
    dataset: Tuple[np.ndarray, np.ndarray], n_targets: int, target_class: int
):
    """
    Given a dataset of form (xs,ys), split it into training/potential poison (1) and potential
    (2) potential targets. Training data consists of all but the last n_target of the
    datapoints for each class (if there are n_targets or fewer points, it consists of all data
    and the target is considered invalid). The nontraining points of the chosen target class
    are returned as potential targets - since the label is known by the caller, the label
    of the target class is not returned.
    """

    target_class = int(target_class)

    total_count_by_class = defaultdict(int)
    curr_count_by_class = defaultdict(int)
    xs, ys = dataset
    for y in ys:
        total_count_by_class[y] += 1

    if total_count_by_class[target_class] <= n_targets:
        raise ValueError(
            f"target_class {target_class} is not a valid target class - fewer "
            f"than {n_targets} data points present"
        )

    xs_train, ys_train, valid_targets = [], [], []
    for x, y in zip(xs, ys):
        if curr_count_by_class[y] < total_count_by_class[y] - n_targets:
            xs_train.append(x)
            ys_train.append(y)
            curr_count_by_class[y] += 1
        elif y == target_class:
            valid_targets.append(x)

    xs_train = np.array(xs_train)
    ys_train = np.array(ys_train)
    valid_targets = np.array(valid_targets)

    return (xs_train, ys_train), valid_targets


def select_poison_indices(
    clean_train_data: Tuple[np.ndarray, np.ndarray],
    n_poison: int,
    poison_images_class: int,
):
    """
    Given a dataset of form (xs,ys), the number of desired poisons, and a potential poison image
    class, return the indices of the subset of data that should be poisoned. Again, because the
    caller knows the labels of the potential poison images, these are not returned.
    """
    xs, ys = clean_train_data
    total_target_class = ys[ys == poison_images_class].shape[0]
    if n_poison > total_target_class:
        raise ValueError(
            f"target_class {poison_images_class} is not a valid target class for "
            f"{n_poison} poisons - not enough data points present."
        )
    return np.where(ys == poison_images_class)[0][:n_poison]
