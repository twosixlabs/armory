"""
Filters here operate on either elements or enumerations of elements
    Those based on enums have 'enum_filter' in the name
"""

import tensorflow as tf


def get_enum_filter_by_index(index: list):
    """
    index must be a list or iterable of integer values
    """
    sorted_index = sorted([int(x) for x in set(index)])
    if len(sorted_index) == 0:
        raise ValueError("The specified dataset 'index' param must be nonempty")
    if sorted_index[0] < 0:
        raise ValueError("The specified dataset 'index' values must be nonnegative")

    def enum_filter_by_index(
        i, element, index_tensor=tf.constant(sorted_index, dtype=tf.int64)
    ):
        i = tf.expand_dims(i, 0)
        out, _ = tf.raw_ops.ListDiff(x=i, y=index_tensor, out_idx=tf.int64)
        return tf.equal(tf.size(out), 0)

    return enum_filter_by_index


def _parse_str_slice(index: str):
    """
    Parse simple slice from string
    """
    index = (
        index.strip().lstrip("[").rstrip("]").strip()
    )  # remove brackets and white space
    tokens = index.split(":")
    if len(tokens) != 2:
        raise ValueError("Slice needs a single ':' character. No fancy slicing.")

    lower, upper = [int(x.strip()) if x.strip() else None for x in tokens]
    if lower is not None and lower < 0:
        raise ValueError(f"slice lower {lower} must be nonnegative")
    if upper is not None and lower is not None and upper <= lower:
        raise ValueError(
            f"slice upper {upper} must be strictly greater than lower {lower}"
        )
    return lower, upper


def get_enum_filter_by_slice(index: str):
    """
    returns the dataset and the indexed size
    """
    lower, upper = _parse_str_slice(index)
    if lower is None:
        lower = 0
    if upper is None:
        upper = tf.int32.max

    def enum_filter_by_slice(i, element, lower=lower, upper=upper):
        return (i >= lower) & (i < upper)

    return enum_filter_by_slice


def get_filter_by_class(class_ids: list, label_key: str):
    """
    Return a function that can be used to filter tf elements
    """
    if len(class_ids) == 0:
        raise ValueError(
            "The specified dataset 'class_ids' param must have at least one value"
        )

    def filter_by_class(
        element,
        label_key=label_key,
        classes_to_keep=tf.constant(class_ids, dtype=tf.int64),
    ):
        y = element[label_key]
        isallowed_array = tf.equal(classes_to_keep, tf.cast(y, tf.int64))
        isallowed = tf.reduce_sum(tf.cast(isallowed_array, tf.int64))
        return tf.greater(isallowed, tf.constant(0, dtype=tf.int64))

    return filter_by_class
