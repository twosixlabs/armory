"""
Generator takes TF Dataset objects and maps them to Armory generators
    Can also be used to produce pytorch data generators
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from art.data_generators import DataGenerator

from armory.logs import log


class ArmoryDataGenerator(DataGenerator):
    """
    Returns batches of numpy data.

    variable_length - if True, returns a 1D object array of arrays for x.
    """

    def __init__(
        self,
        generator,
        size,
        epochs,
        batch_size,
        preprocessing_fn=None,
        label_preprocessing_fn=None,
        variable_length=False,
        variable_y=False,
        context=None,
    ):
        super().__init__(size, batch_size)
        self.preprocessing_fn = preprocessing_fn
        self.label_preprocessing_fn = label_preprocessing_fn
        self.generator = generator

        self.epochs = epochs
        self.samples_per_epoch = size

        # drop_remainder is False
        self.batches_per_epoch = self.samples_per_epoch // batch_size + bool(
            self.samples_per_epoch % batch_size
        )

        self.variable_length = variable_length
        self.variable_y = variable_y
        if self.variable_length or self.variable_y:
            self.current = 0

        self.context = context

    @staticmethod
    def np_1D_object_array(x_list):
        """
        Take a list of single-element batches and return as a numpy 1D object array

        Similar to np.stack, but designed to handle variable-length elements
        """
        x = np.empty((len(x_list),), dtype=object)
        for i in range(len(x_list)):
            x[i] = x_list[i][0]
        return x

    def get_batch(self) -> Tuple[np.ndarray, Union[np.ndarray, List]]:
        if self.variable_length or self.variable_y:
            # build the batch
            x_list, y_list = [], []
            for i in range(self.batch_size):
                x_i, y_i = next(self.generator)
                x_list.append(x_i)
                y_list.append(y_i)
                self.current += 1
                # handle end of epoch partial batches
                if self.current == self.samples_per_epoch:
                    self.current = 0
                    break

            if self.variable_length:
                if isinstance(x_list[0], dict):
                    # Translate a list of dicts into a dict of arrays
                    x = {}
                    for k in x_list[0].keys():
                        x[k] = self.np_1D_object_array([x_i[k] for x_i in x_list])
                elif isinstance(x_list[0], tuple):
                    # Translate a list of tuples into a tuple of arrays
                    x = tuple(self.np_1D_object_array(i) for i in zip(*x_list))
                else:
                    x = self.np_1D_object_array(x_list)
            else:
                x = np.vstack(x_list)

            if self.variable_y:
                if isinstance(y_list[0], dict):
                    # Store y as a list of dicts
                    y = y_list
                elif isinstance(y_list[0], tuple):
                    # Translate a list of tuples into a tuple of arrays
                    y = tuple(self.np_1D_object_array(i) for i in zip(*y_list))
                else:
                    y = self.np_1D_object_array(y_list)
            else:
                if isinstance(y_list[0], dict):
                    y = {}
                    for k in y_list[0].keys():
                        y[k] = np.hstack([y_i[k] for y_i in y_list])
                elif isinstance(y_list[0], tuple):
                    y = tuple(np.hstack(i) for i in zip(*y_list))
                else:
                    y = np.hstack(y_list)
        else:
            log.debug(next(self.generator))
            x, y = next(self.generator)

        if self.label_preprocessing_fn:
            y = self.label_preprocessing_fn(x, y)

        if self.preprocessing_fn:
            # Apply preprocessing to multiple inputs as needed
            if isinstance(x, dict):
                x = {k: self.preprocessing_fn(v) for (k, v) in x.items()}
            elif isinstance(x, tuple):
                x = tuple(self.preprocessing_fn(i) for i in x)
            else:
                x = self.preprocessing_fn(x)
        return x, y

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def __len__(self):
        return self.batches_per_epoch * self.epochs


# TODO:  David tried to refactor this at one point but ran into some issues
# but probably not insurmounting
class EvalGenerator(DataGenerator):
    """
    Wraps a specified number of batches in a DataGenerator to allow for evaluating on
    part of a dataset when running through a scenario
    """

    def __init__(self, armory_generator, num_eval_batches):
        if not isinstance(armory_generator, ArmoryDataGenerator):
            raise ValueError(f"{armory_generator} is not of type ArmoryDataGenerator")
        super().__init__(armory_generator.batch_size, armory_generator.batch_size)
        self.armory_generator = armory_generator
        self.num_eval_batches = num_eval_batches
        self.batches_processed = 0
        # This attr is only used by ucf video scenarios that involve finetuning. It
        # must be set to enable check runs.
        self.batches_per_epoch = 1
        self.context = armory_generator.context

    def get_batch(self) -> (np.ndarray, np.ndarray):
        if self.batches_processed == self.num_eval_batches:
            raise StopIteration()
        batch = self.armory_generator.get_batch()
        self.batches_processed += 1
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def __len__(self):
        return self.num_eval_batches


def generator_from_dataset(
    dataset_info: dict,
    dataset: dict,
    framework: str = "numpy",
    split: str = "train",
    batch_size: int = 1,
    epochs: int = 1,
    preprocessing_fn: Callable = None,
    label_preprocessing_fn: Callable = None,
    variable_length: bool = False,
    variable_y: bool = False,
    shuffle_files: bool = False,
    as_supervised: bool = True,
    supervised_xy_keys: Optional[tuple] = None,
    lambda_map: Callable = None,
    context=None,
    class_ids=None,
    index=None,
) -> Union[ArmoryDataGenerator, tf.data.Dataset]:

    ds = dataset[split]

    if not as_supervised:
        try:
            x_key, y_key = supervised_xy_keys
        except (TypeError, ValueError):
            raise ValueError(
                f"When as_supervised=False, supervised_xy_keys must be a (x_key, y_key)"
                f" tuple, not {supervised_xy_keys}"
            )
        for key in [x_key, y_key]:
            if not (isinstance(key, str) or isinstance(key, tuple)):
                raise ValueError(
                    f"supervised_xy_keys must be a tuple of strings or a tuple of tuple of strings"
                    f" not {type(x_key), type(y_key)}"
                )
        if isinstance(x_key, tuple):
            if isinstance(y_key, tuple):
                raise ValueError(
                    "Only one of (x_key, y_key) can be a tuple while the other must be a string."
                )
            for k in x_key:
                if not (isinstance(k, str)):
                    raise ValueError(
                        "supervised_xy_keys must be a tuple of strings or a tuple of tuple of strings"
                    )
            ds = ds.map(lambda x: (tuple(x[k] for k in x_key), x[y_key]))
        elif isinstance(y_key, tuple):
            for k in y_key:
                if not (isinstance(k, str)):
                    raise ValueError(
                        "supervised_xy_keys must be a tuple of strings or a tuple of tuple of strings"
                    )
            ds = ds.map(lambda x: (x[x_key], tuple(x[k] for k in y_key)))
        else:
            ds = ds.map(lambda x: (x[x_key], x[y_key]))

    # TODO:  Remove , thesea are just notes
    #  this shows up in so2stat call
    if lambda_map is not None:
        ds = ds.map(lambda_map)

    dataset_size = dataset_info.splits[split].num_examples

    # Add class-based filtering
    if class_ids is not None:
        if split == "train":
            log.warning(
                "Filtering by class entails iterating over the whole dataset and thus "
                "can be very slow if using the 'train' split"
            )

        # TODO: Why not use TFDS filter ?? (https://www.tensorflow.org/datasets/decode#only_decode_a_sub-set_of_the_features)

        # TODO: Remove when done
        #  Issue with filtering, to know the len you have to iterate the entire dataset, doesn't appear in metadata
        #  Filter by index is fast, filter by stringslice is very slow
        #  Figure out why we need the full dataset size
        #  Add to ArmoryDataGenerator -> add_filter that removes samples at execution time based on filter
        # if isinstance(class_ids, list):
        #     ds, dataset_size = filter_by_class(ds, class_ids=class_ids)
        # elif isinstance(class_ids, int):
        #     ds, dataset_size = filter_by_class(ds, class_ids=[class_ids])
        # else:
        #     raise ValueError(
        #         f"class_ids must be a list, int, or None, not {type(class_ids)}"
        #     )

    # # Add index-based filtering
    # if isinstance(index, list):
    #     ds, dataset_size = filter_by_index(ds, index, dataset_size)
    # elif isinstance(index, str):
    #     ds, dataset_size = filter_by_str_slice(ds, index, dataset_size)
    # elif index is not None:
    #     raise ValueError(f"index must be a list, str, or None, not {type(index)}")

    ds = ds.repeat(epochs)
    # TODO: Why is this here since builder does this already??
    #  shuffle files is a part of original builder but not during execution
    #  maybe not needed if we shuffle the files at build time
    if shuffle_files:
        ds = ds.shuffle(batch_size * 10, reshuffle_each_iteration=True)
    if variable_length or variable_y and batch_size > 1:
        ds = ds.batch(1, drop_remainder=False)
    else:
        ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    log.info(f"ds: {ds}")

    if framework == "numpy":
        ds = tfds.as_numpy(ds)
        log.debug(f"Numpy ds: {ds}")
        generator = ArmoryDataGenerator(
            iter(ds),
            size=dataset_size,
            batch_size=batch_size,
            epochs=epochs,
            preprocessing_fn=preprocessing_fn,
            label_preprocessing_fn=label_preprocessing_fn,
            variable_length=bool(variable_length and batch_size > 1),
            variable_y=bool(variable_y and batch_size > 1),
            context=context,
        )

    elif framework == "tf":
        generator = ds

    elif framework == "pytorch":
        from armory.datasets.pytorch_loader import get_pytorch_data_loader

        return get_pytorch_data_loader(ds)
    else:
        raise ValueError(
            f"`framework` must be one of ['tf', 'pytorch', 'numpy']. Found {framework}"
        )

    log.debug(f"generator: {generator}")
    return generator
