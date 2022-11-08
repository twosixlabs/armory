"""
Example:

from datasets import load, generator
info, ds = load.load("digit")
gen = generator.Generator(info, ds, element_map = lambda z: (generator.audio_to_canon(z["audio"]), z["label"]))
x, y = next(gen)

info, ds = load.load("mnist")
gen = generator.Generator(info, ds, element_map = lambda z: (generator.image_to_canon(z["image"]), z["label"]), framework="tf", batch_size=5)
x, y = next(gen)

"""

import tensorflow as tf

# NOTE: currently does not extend art.data_generators.DataGenerator
#     which is necessary for using ART `fit_generator` method
class ArmoryDataGenerator:
    """
    Returns batches of numpy data

    Raw = no preprocessing

    ds_dict = dataset dictionary without split selected

    num_eval_batches - if not None, the number of batches to grab

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
        num_eval_batches: int = None,
        framework: str = "numpy",
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
        if num_eval_batches is not None and int(num_eval_batches) < 1:
            raise ValueError(
                f"num_eval_batches must be None or a positive integer, not {num_eval_batches}"
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

        if framework == "tf":
            iterator = iter(ds)
        elif framework == "numpy":
        # ds = tfds.as_numpy(ds)  # TODO: this or tfds.as_numpy_iterator() ?
            iterator = ds.as_numpy_iterator()
        else:  # torch
            # SEE: tfds.as_numpy
            # https://github.com/tensorflow/datasets/blob/v4.7.0/tensorflow_datasets/core/dataset_utils.py#L141-L176
            raise NotImplementedError(f"framework {framework}")
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
            framework=framework,
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

def image_to_canon(image, resize=None, target_dtype=tf.float32, input_type="uint8"):
    """
    TFDS Image feature uses (height, width, channels)
    """
    if input_type == "uint8":
        scale = 255.0
    else:
        raise NotImplementedError(f"Currently only supports uint8, not {input_type}")
    image = tf.cast(image, target_dtype)
    image = image / scale
    if resize is not None:
        resize = tuple(size)
        if len(resize) != 2:
            raise ValueError(f"resize must be None or a 2-tuple, not {resize}")
        image = tf.image.resize(image, resize)
    return image


def audio_to_canon(audio, resample=None, target_dtype=tf.float32, input_type="int16"):
    """
    Note: input_type is the scale of the actual data
        TFDS typically converts to tf.inf64, which is not helpful in this case
    """
    if input_type == "int16":
        scale = 2**15
    else:
        raise NotImplementedError(f"Currently only supports uint8, not {input_type}")
    audio = tf.cast(audio, target_dtype)
    audio = audio / scale
    if resample is not None:
        raise NotImplementedError(f"resampling not currently supported")
    return audio


def video_to_canon(video, resize=None, target_dtype=tf.float32, input_type="uint8", max_frames: int = None):
    """
    TFDS Video feature uses (num_frames, height, width, channels)
    """
    if input_type == "uint8":
        scale = 255.0
    else:
        raise NotImplementedError(f"Currently only supports uint8, not {input_type}")
    
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        video = video[:max_frames]
    video = tf.cast(video, target_dtype)
    video = video / scale
    if resize is not None:
        raise NotImplementedError(f"resizing video")
    return video
