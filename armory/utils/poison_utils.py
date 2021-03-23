import numpy as np


def load_in_memory(data):
    x_all, y_all = [], []
    for x, y in data:
        x_all.append(x)
        y_all.append(y)
    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return x_all, y_all
