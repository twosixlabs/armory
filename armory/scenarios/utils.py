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
