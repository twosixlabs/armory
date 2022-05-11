"""
CSV reading fill-in to not require pandas dependency.

A common pattern in adversarial datasets is:
    import pandas
    pandas.read_csv(path, header=None).to_numpy().astype("float32")

This file is meant to replace that as follows:
    import pandas_proxy
    pandas_proxy.read_csv_to_numpy_float32(path, header=None)
"""

import csv

import numpy as np


def read_csv_to_numpy_float32(path, header=None) -> "np.array":
    if header is not None:
        raise NotImplementedError("non-None header not supported")

    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(tuple((float(x) for x in row)))
    return np.array(rows, dtype=np.float32)
