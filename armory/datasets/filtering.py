def _parse_token(token: str):
    """
    Token from parse_split index

    Return parsed token
    """
    if not token:
        raise ValueError("empty token found")

    left = token.find("[")
    if left == -1:
        return token

    right = token.rfind("]")
    if right != len(token) - 1:
        raise ValueError(f"could not parse token {token} - mismatched brackets")
    name = token[:left]
    index = token[left + 1 : right]  # remove brackets
    if not name:
        raise ValueError(f"empty split name: {token}")
    if not index:
        raise ValueError(f"empty index found: {token}")
    if index.count(":") == 2:
        raise NotImplementedError(f"slice 'step' not enabled: {token}")
    elif index == "[]":
        raise ValueError(f"empty list index found: {token}")

    if re.match(r"^\d+$", index):
        # single index
        i = int(index)
        return f"{name}[{i}:{i+1}]"
    elif re.match(r"^\[\d+(\s*,\s*\d+)*\]$", index):
        # list index of nonnegative integer indices
        # out-of-order and duplicate indices are allowed
        index_list = json.loads(index)
        token_list = [f"{name}[{i}:{i+1}]" for i in index_list]
        if not token_list:
            raise ValueError
        return "+".join(token_list)

    return token


def parse_split_index(split: str):
    """
    Take a TFDS split argument and rewrite index arguments such as:
        test[10] --> test[10:11]
        test[[1, 5, 7]] --> test[1:2]+test[5:6]+test[7:8]
    """
    if not isinstance(split, str):
        raise ValueError(f"split must be str, not {type(split)}")
    if not split.strip():
        raise ValueError("split cannot be empty")

    tokens = split.split("+")
    tokens = [x.strip() for x in tokens]
    output_tokens = [_parse_token(x) for x in tokens]
    return "+".join(output_tokens)


def filter_by_index(dataset: "tf.data.Dataset", index: list, dataset_size: int):
    """
    index must be a list or iterable of integer values

    returns the dataset and the indexed size
    """
    log.info(f"Filtering dataset to the following indices: {index}")
    dataset_size = int(dataset_size)
    sorted_index = sorted([int(x) for x in set(index)])
    if len(sorted_index) == 0:
        raise ValueError("The specified dataset 'index' param must be nonempty")
    if sorted_index[0] < 0:
        raise ValueError("The specified dataset 'index' values must be nonnegative")
    if sorted_index[-1] >= dataset_size:
        raise ValueError(
            f"The specified dataset 'index' values exceed dataset size {dataset_size}"
        )
    num_valid_indices = len(sorted_index)

    index_tensor = tf.constant(sorted_index, dtype=tf.int64)

    def enum_index(i, x):
        i = tf.expand_dims(i, 0)
        out, _ = tf.raw_ops.ListDiff(x=i, y=index_tensor, out_idx=tf.int64)
        return tf.equal(tf.size(out), 0)

    return dataset.enumerate().filter(enum_index).map(lambda i, x: x), num_valid_indices


def filter_by_class(dataset: "tf.data.Dataset", class_ids: Union[list, int]):
    """
    class_ids must be an int or list of ints

    returns the dataset filtered by class id, keeping elements with label in class_ids
    """
    log.info(f"Filtering dataset to the following class IDs: {class_ids}")
    if len(class_ids) == 0:
        raise ValueError(
            "The specified dataset 'class_ids' param must have at least one value"
        )

    def _filter_by_class(x, y, classes_to_keep=tf.constant(class_ids, dtype=tf.int64)):
        isallowed_array = tf.equal(classes_to_keep, tf.cast(y, tf.int64))
        isallowed = tf.reduce_sum(tf.cast(isallowed_array, tf.int64))
        return tf.greater(isallowed, tf.constant(0, dtype=tf.int64))

    filtered_ds = dataset.filter(_filter_by_class)

    if tf.executing_eagerly():
        filtered_ds_size = int(filtered_ds.reduce(0, lambda x, _: x + 1).numpy())
    else:
        filtered_ds_size = len(list(tfds.as_numpy(filtered_ds)))

    if filtered_ds_size == 0:
        raise ValueError(
            "All elements of dataset were removed. Please ensure the specified class_ids appear in the dataset"
        )

    return filtered_ds, filtered_ds_size


def parse_str_slice(index: str):
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


def filter_by_str_slice(dataset: "tf.data.Dataset", index: str, dataset_size: int):
    """
    returns the dataset and the indexed size
    """
    lower, upper = parse_str_slice(index)
    if lower is None:
        lower = 0
    if upper is None:
        upper = dataset_size
    if lower >= dataset_size:
        raise ValueError(f"lower {lower} must be less than dataset_size {dataset_size}")
    if upper > dataset_size:
        upper = dataset_size
    indexed_size = upper - lower

    def slice_index(i, x):
        return (i >= lower) & (i < upper)

    return dataset.enumerate().filter(slice_index).map(lambda i, x: x), indexed_size
