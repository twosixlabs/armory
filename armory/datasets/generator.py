"""

"""

import tensorflow as tf

# NOTE: currently does not extend art.data_generators.DataGenerator
#     which is necessary for using ART `fit_generator` method
class ArmoryRawDataGenerator:
    """
    Returns batches of numpy data

    Raw = no preprocessing

    ds_dict = dataset dictionary without split selected

    num_eval_batches - if not None, the number of batches to grab
    """

    def __init__(
        self,
        info,
        ds_dict: dict,
        split: str = "test",
        batch_size: int = 1,
        drop_remainder: bool = False,
        epochs: int = 1,
        num_eval_batches: int = None,
    ):
        if split not in info.splits:
            raise ValueError(f"split {split} not in info.splits {list(info.splits)}")
        if split not in ds_dict:
            raise ValueError(f"split {split} not in ds_dict keys {list(ds_dict)}")
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be a positive integer, not {batch_size}")
        if int(epochs) < 1:
            raise ValueError(f"epochs must be a positive integer, not {epochs}")
        if num_eval_batches is not None and int(num_eval_batches) < 1:
            raise ValueError(
                f"num_eval_batches must be None or a positive integer, not {num_eval_batches}"
            )
        size = info.splits[split].num_examples
        batch_size = int(batch_size)
        batches_per_epoch = size // batch_size
        if not drop_remainder:
            batches_per_epoch += bool(size % batch_size)
        ds = ds_dict[split]
        if num_eval_batches is not None:
            num_eval_batches = int(num_eval_batches)
            if num_eval_batches > batches_per_epoch:
                raise ValueError(
                    f"num_eval_batches {num_eval_batches} cannot be greater than batches_per_epoch {batches_per_epoch}"
                )
            size = num_eval_batches * batch_size
            batches_per_epoch = num_eval_batches
            ds = ds.take(size)

        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        # ds = ds.cache()
        # ds = tfds.as_numpy(ds)  # TODO: this or tfds.as_numpy_iterator() ?
        iterator = ds.as_numpy_iterator()
        # TODO: figure out how to keep the following error from happening (or prevent it from printing to screen):
        # 2022-11-04 22:08:44.849705: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
        # Likely do to the https://www.tensorflow.org/datasets/api_docs/python/tfds/ReadConfig
        # ReadConfig having try_autocache: bool = True in builder.as_dataset method https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetBuilder
        # Alternatively, use https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints to limit C++ logging as well
        #   This could be put into `armory.logs`

        self._set_params(
            iterator=iterator,
            split=split,
            size=size,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            drop_remainder=drop_remainder,
            epochs=epochs,
            num_eval_batches=num_eval_batches,
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


# TODO: Armory Generator that enables preprocessing and mapping into tuples
