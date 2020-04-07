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
        raise ValueError(f"{y} and {y_pred} accuracy cannot be computed")


def linf(x, x_adv):
    """
    Return the L infinity norm over a batch of inputs as a float
    """
    # uint8 inputs will return incorrect results, so cast to float
    diff = np.abs(x.astype(float) - x_adv.astype(float))
    return list(np.max(diff, tuple(range(1, diff.ndim))))
