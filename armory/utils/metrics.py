"""
Metrics for scenarios

Outputs are lists of python variables amenable to JSON serialization:
    e.g., bool, int, float
    numpy data types and tensors generally fail to serialize
"""

import numpy as np


def categorical_accuracy(y, y_pred):
    """
    Return the categorical accuracy of the predictions
    """
    if y.shape == y_pred.shape:
        return [bool(x) for x in list(y == y_pred)]
    elif y.ndim + 1 == y_pred.ndim:
        return [bool(x) for x in list(y == np.argmax(y_pred, axis=1))]
    else:
        raise ValueError(f"{y} and {y_pred} have mismatched dimensions")


def linf(x, x_adv):
    """
    Return the L infinity norm over a batch of inputs as a float
    """
    # uint8 inputs will return incorrect results, so cast to float
    diff = np.abs(x.astype(float) - x_adv.astype(float))
    return list(np.max(diff, tuple(range(1, diff.ndim))))


class AverageMeter:
    """
    Computes and stores the average and current value

    Taken from https://github.com/craston/MARS/blob/master/utils.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
