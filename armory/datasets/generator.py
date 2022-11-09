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

import tensorflow as tf
import tensorflow_datasets as tfds


# NOTE: currently does not extend art.data_generators.DataGenerator
#     which is necessary for using ART `fit_generator` method
class ArmoryDataGenerator:
    """
    Returns batches of numpy data

    Raw = no preprocessing

    ds_dict = dataset dictionary without split selected

    num_batches - if not None, the number of batches to grab

    element_filter - predicate of which elements to keep; occurs prior to mapping
        Note: size computations will be wrong when filtering is applied
    element_map - function that takes a dataset element (dict) and maps to new element
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
        element_filter: callable = None,
        element_map: callable = None,
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

        size = info.splits[split].num_examples
        batch_size = int(batch_size)
        batches_per_epoch = size // batch_size
        if not drop_remainder:
            batches_per_epoch += bool(size % batch_size)

        ds = ds_dict[split]
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
            # TODO: maybe set to size of data shards? How do we find this info?
            # info.splits['train'].num_examples / info.splits['train'].num_shards

            buffer_size = batch_size * 10
            ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True)
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
        )

    def _set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)

    def __len__(self):
        return self.batches_per_epoch * self.epochs
