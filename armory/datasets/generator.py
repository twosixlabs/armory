"""
Example:

from datasets import load, generator, preprocessing
info, ds = load.load("digit")
gen = generator.ArmoryDataGenerator(info, ds, element_map = lambda z: (preprocessing.audio_to_canon(z["audio"]), z["label"]))
x, y = next(gen)

info, ds = load.load("mnist")
gen = generator.ArmoryDataGenerator(info, ds, element_map = lambda z: (preprocessing.image_to_canon(z["image"]), z["label"]), framework="tf", batch_size=5)
x, y = next(gen)

"""

from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
import math


class ArmoryDataGenerator:
    """
    Returns batches of numpy data

    Raw = no preprocessing

    ds_dict - dataset dictionary without split selected

    Without a split specified, load.load will return an info and a ds_dict:
        info, ds_dict = load.load("mnist")

    If a split is specified in load.load, then a ds will be returned
        info, ds = load.load("mnist", split="test")

    We use the ds_dict here, because otherwise we can't grab split information from info

    num_batches - if not None, the number of batches to grab

    index_filter - predicate of which indexes to keep
    element_filter - predicate of which elements to keep; occurs prior to mapping
        Note: size computations will be wrong when filtering is applied
    element_map - function that takes a dataset element (dict) and maps to new element

    key_map - dict that maps element keys to scenario keys such as `x` and `y`
        Example: {"image": "x", "label": "y"}
        if None, no mapping is done

    as_tuple - if None, iterator produces batches of dicts
        If not None, iterator produces a tuple based on the given set of keys
    """

    FRAMEWORKS = ("tf", "numpy", "torch")

    def __init__(
        self,
        info,
        ds_dict: dict,
        split: str = "test",
        batch_size: int = 1,
        drop_remainder: bool = False,
        epochs: int = 1,
        num_batches: int = None,
        framework: str = "numpy",
        shuffle_elements: bool = False,
        index_filter: callable = None,
        element_filter: callable = None,
        element_map: callable = None,
        key_map=None,
        as_tuple: Tuple[str] = None,
    ):
        if split not in info.splits:
            raise ValueError(f"split {split} not in info.splits {list(info.splits)}")
        if split not in ds_dict:
            raise ValueError(f"split {split} not in ds_dict keys {list(ds_dict)}")
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be a positive integer, not {batch_size}")
        if int(epochs) < 1:
            raise ValueError(f"epochs must be a positive integer, not {epochs}")
        if num_batches is not None and int(num_batches) < 1:
            raise ValueError(
                f"num_batches must be None or a positive integer, not {num_batches}"
            )
        if framework not in self.FRAMEWORKS:
            raise ValueError(f"framework {framework} not in {self.FRAMEWORKS}")
        if key_map is not None:
            for k, v in key_map.items():
                for i in (k, v):
                    if not isinstance(i, str):
                        raise ValueError(f"{i} in key_map is not a str")
            # TODO: key mapping from dict to tuples, etc.
            raise NotImplementedError("key_map argument")
        if as_tuple is not None:
            if isinstance(as_tuple, str):
                raise ValueError(f"as_tuple must be None or a tuple of str, not str")
            as_tuple = tuple(as_tuple)
            for k in as_tuple:
                if not isinstance(k, str):
                    raise ValueError(f"item {k} in as_tuple is not a str")
        size = info.splits[split].num_examples
        batch_size = int(batch_size)
        batches_per_epoch = size // batch_size
        if not drop_remainder:
            batches_per_epoch += bool(size % batch_size)

        ds = ds_dict[split]
        if index_filter is not None:
            ds = ds.enumerate().filter(index_filter).map(lambda i, x: x)
        if element_filter is not None:
            ds = ds.filter(element_filter)
        if element_map is not None:
            ds = ds.map(element_map)

        if num_batches is not None:
            num_batches = int(num_batches)
            if num_batches > batches_per_epoch:
                raise ValueError(
                    f"num_batches {num_batches} cannot be greater than batches_per_epoch {batches_per_epoch}"
                )
            size = num_batches * batch_size
            batches_per_epoch = num_batches
            ds = ds.take(size)

        if epochs > 1:
            ds = ds.repeat(epochs)
        if shuffle_elements:
            # https://www.tensorflow.org/datasets/performances#caching_the_dataset
            # for true random, set buffer_size to dataset size
            #     for large datasets, this is too large, therefore use shard size
            # set shuffle buffer to the number of elements in a shard
            # shuffle_files will shuffle between shard, this will shuffle within shards
            #     this won't be true random, but will be sufficiently close
            examples_per_split = math.ceil(size / info.splits[split].num_shards)
            ds = ds.shuffle(examples_per_split, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        # ds = ds.cache()

        if framework == "tf":
            iterator = iter(ds)
        elif framework == "numpy":
            # ds = tfds.as_numpy(ds)  # TODO: this or tfds.as_numpy_iterator() ?
            ds = tfds.as_numpy(ds)  # works in non-eager mode as well
            iterator = iter(ds)
            # iterator = ds.as_numpy_iterator()  # fails in non-eager mode
        else:  # torch
            # SEE: tfds.as_numpy
            # https://github.com/tensorflow/datasets/blob/v4.7.0/tensorflow_datasets/core/dataset_utils.py#L141-L176
            raise NotImplementedError(f"framework {framework}")

        self._set_params(
            iterator=iterator,
            split=split,
            size=size,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            drop_remainder=drop_remainder,
            epochs=epochs,
            num_batches=num_batches,
            framework=framework,
            shuffle_elements=shuffle_elements,
            element_filter=element_filter,
            element_map=element_map,
            key_map=key_map,
            as_tuple=as_tuple,
        )

    def _set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_as_tuple(self, as_tuple: Tuple[str]):
        if as_tuple is not None:
            if isinstance(as_tuple, str):
                raise ValueError(f"as_tuple must be None or a tuple of str, not str")
            as_tuple = tuple(as_tuple)
            for k in as_tuple:
                if not isinstance(k, str):
                    raise ValueError(f"item {k} in as_tuple is not a str")
        self.as_tuple = as_tuple

    def __iter__(self):
        return self

    def __next__(self):
        element = next(self.iterator)
        if self.key_map is not None:
            element = {new_k: element[k] for k, new_k in key_map.items()}
        if self.as_tuple is not None:
            element = tuple(element[k] for k in self.as_tuple)
        return element

    def __len__(self):
        return self.batches_per_epoch * self.epochs
