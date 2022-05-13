import os
from typing import Callable, Union, Tuple, List, Optional

import numpy as np
from armory.logs import log

import torch
import tensorflow as tf
import tensorflow_datasets as tfds
from art.data_generators import DataGenerator

from armory.data.utils import (
    download_verify_dataset_cache,
    _read_validate_scenario_config,
    add_checksums_dir,
)
from armory import paths


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
