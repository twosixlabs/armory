"""
Adversarial datasets
"""

from typing import Callable

import numpy as np
import tensorflow as tf

from armory.data import datasets
from armory.data.adversarial import (  # noqa: F401
    imagenet_adversarial as IA,
    librispeech_adversarial as LA,
    resisc45_densenet121_univpatch_and_univperturbation_adversarial_224x224,
    ucf101_mars_perturbation_and_patch_adversarial_112x112,
    gtsrb_bh_poison_micronnet,
    apricot_dev,
)


# The APRICOT dataset uses class ID 12 to correspond to adversarial patches. Since this
# number may correspond to real classes in other datasets, we convert this label 12 in the
# APRICOT dataset to the ADV_PATCH_MAGIC_NUMBER_LABEL_ID. We choose a negative integer
# since it is unlikely that such a number represents the ID of a class in another dataset
ADV_PATCH_MAGIC_NUMBER_LABEL_ID = -10


def imagenet_adversarial(
    split_type: str = "adversarial",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    clean_key: str = "clean",
    adversarial_key: str = "adversarial",
    targeted: bool = False,
    shuffle_files: bool = False,
) -> datasets.ArmoryDataGenerator:
    """
    ILSVRC12 adversarial image dataset for ResNet50

    ProjectedGradientDescent
        Iterations = 10
        Max perturbation epsilon = 8
        Attack step size = 2
        Targeted = True
    """
    if clean_key != "clean":
        raise ValueError(f"{clean_key} != 'clean'")
    if adversarial_key != "adversarial":
        raise ValueError(f"{adversarial_key} != 'adversarial'")
    if targeted:
        raise ValueError(f"{adversarial_key} is not a targeted attack")

    return datasets._generator_from_tfds(
        "imagenet_adversarial:1.1.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=imagenet_canonical_preprocessing,
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda x, y: ((x[clean_key], x[adversarial_key]), y),
    )


class ImagenetContext:
    def __init__(self):
        self.default_float = np.float32
        self.quantization = 255
        self.x_dimensions = (None, 224, 224, 3)


imagenet_context = ImagenetContext()


def imagenet_canonical_preprocessing(batch):
    if batch.ndim != len(imagenet_context.x_dimensions):
        raise ValueError(
            f"input batch dim {batch.ndim} != {len(imagenet_context.x_dimensions)}"
        )
    assert batch.dtype == np.uint8
    assert batch.shape[1:] == imagenet_context.x_dimensions[1:]
    assert batch.dtype == np.uint8

    batch = batch.astype(imagenet_context.default_float) / imagenet_context.quantization
    assert batch.dtype == imagenet_context.default_float
    assert batch.max() <= 1.0
    assert batch.min() >= 0.0

    return batch


def librispeech_adversarial(
    split_type: str = "adversarial",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = datasets.librispeech_dev_clean_dataset_canonical_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    clean_key: str = "clean",
    adversarial_key: str = "adversarial_perturbation",
    targeted: bool = False,
    shuffle_files: bool = False,
) -> datasets.ArmoryDataGenerator:
    """
    Adversarial dataset based on Librispeech-dev-clean including clean,
    Universal Perturbation using PGD, and PGD.

    split_type - one of ("adversarial")

    returns:
        Generator
    """
    if clean_key != "clean":
        raise ValueError(f"{clean_key} != 'clean'")
    adversarial_keys = ("adversarial_perturbation", "adversarial_univperturbation")
    if adversarial_key not in adversarial_keys:
        raise ValueError(f"{adversarial_key} not in {adversarial_keys}")
    if targeted:
        raise ValueError(f"{adversarial_key} is not a targeted attack")

    return datasets._generator_from_tfds(
        "librispeech_adversarial:1.1.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("audio", "label"),
        variable_length=bool(batch_size > 1),
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda x, y: ((x[clean_key], x[adversarial_key]), y),
    )


def resisc45_adversarial_224x224(
    split_type: str = "adversarial",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    clean_key: str = "clean",
    adversarial_key: str = "adversarial_univperturbation",
    targeted: bool = False,
    shuffle_files: bool = False,
) -> datasets.ArmoryDataGenerator:
    """
    resisc45 Adversarial Dataset of size (224, 224, 3),
    including clean, adversarial universal perturbation, and
    adversarial patched
    """
    if clean_key != "clean":
        raise ValueError(f"{clean_key} != 'clean'")
    adversarial_keys = ("adversarial_univpatch", "adversarial_univperturbation")
    if adversarial_key not in adversarial_keys:
        raise ValueError(f"{adversarial_key} not in {adversarial_keys}")
    if targeted:
        if adversarial_key == "adversarial_univperturbation":
            raise ValueError("adversarial_univperturbation is not a targeted attack")

        def lambda_map(x, y):
            return (
                (x[clean_key], x[adversarial_key]),
                (y[clean_key], y[adversarial_key]),
            )

    else:

        def lambda_map(x, y):
            return (x[clean_key], x[adversarial_key]), y[clean_key]

    return datasets._generator_from_tfds(
        "resisc45_densenet121_univpatch_and_univperturbation_adversarial224x224:1.0.2",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("images", "labels"),
        variable_length=False,
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda_map,
    )


def ucf101_adversarial_112x112(
    split_type: str = "adversarial",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    clean_key: str = "clean",
    adversarial_key: str = "adversarial_perturbation",
    targeted: bool = False,
    shuffle_files: bool = False,
) -> datasets.ArmoryDataGenerator:
    """
    UCF 101 Adversarial Dataset of size (112, 112, 3),
    including clean, adversarial perturbed, and
    adversarial patched

    DataGenerator returns batches of ((x_clean, x_adversarial), y)
    """
    if clean_key != "clean":
        raise ValueError(f"{clean_key} != 'clean'")
    adversarial_keys = ("adversarial_patch", "adversarial_perturbation")
    if adversarial_key not in adversarial_keys:
        raise ValueError(f"{adversarial_key} not in {adversarial_keys}")
    if targeted:
        if adversarial_key == "adversarial_perturbation":
            raise ValueError("adversarial_perturbation is not a targeted attack")

        def lambda_map(x, y):
            return (
                (x[clean_key], x[adversarial_key]),
                (y[clean_key], y[adversarial_key]),
            )

    else:

        def lambda_map(x, y):
            return (x[clean_key], x[adversarial_key]), y[clean_key]

    return datasets._generator_from_tfds(
        "ucf101_mars_perturbation_and_patch_adversarial112x112:1.1.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("videos", "labels"),
        variable_length=bool(batch_size > 1),
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda_map,
    )


def gtsrb_poison(
    split_type: str = "poison",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    clean_key: str = None,
    adversarial_key: str = None,
    shuffle_files: bool = False,
) -> datasets.ArmoryDataGenerator:
    """
    German traffic sign poison dataset of size (48, 48, 3),
    including only poisoned data

    DataGenerator returns batches of (x_poison, y)
    """
    return datasets._generator_from_tfds(
        "gtsrb_bh_poison_micronnet:1.0.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("image", "label"),
        variable_length=bool(batch_size > 1),
        shuffle_files=False,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda x, y: (x, y),
    )


class ApricotContext:
    def __init__(self):
        self.default_float = np.float32
        self.quantization = 255
        self.x_dimensions = (None, None, None, 3)


apricot_context = ApricotContext()


def apricot_canonical_preprocessing(batch):
    if batch.ndim != len(apricot_context.x_dimensions):
        raise ValueError(
            f"input batch dim {batch.ndim} != {len(apricot_context.x_dimensions)}"
        )
    assert batch.dtype == np.uint8
    assert batch.shape[3] == apricot_context.x_dimensions[3]

    batch = batch.astype(apricot_context.default_float) / apricot_context.quantization

    assert batch.dtype == apricot_context.default_float
    assert batch.max() <= 1.0
    assert batch.min() >= 0.0

    return batch


def apricot_dev_adversarial(
    split_type: str = "adversarial",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = apricot_canonical_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
) -> datasets.ArmoryDataGenerator:
    if batch_size != 1:
        raise NotImplementedError("Currently working only with batch size = 1")

    # The apricot dataset uses 12 as the label for adversarial patches, which may be used for
    # meaningful categories for other datasets. This method is applied as a lambda_map to convert
    #  this label from 12 to the ADV_PATCH_MAGIC_NUMBER_LABEL_ID -- we choose a negative integer
    #  for the latter since it is unlikely that such a number represents the ID of a class in
    # another dataset
    raw_adv_patch_category_id = 12

    def replace_magic_val(data, raw_val, transformed_val, sub_key):
        rhs = data[sub_key]
        data[sub_key] = tf.where(
            tf.equal(rhs, raw_val),
            tf.ones_like(rhs, dtype=tf.int64) * transformed_val,
            rhs,
        )
        return data

    return datasets._generator_from_tfds(
        "apricot_dev:1.0.1",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("image", "objects"),
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda x, y: (
            x,
            replace_magic_val(
                y, raw_adv_patch_category_id, ADV_PATCH_MAGIC_NUMBER_LABEL_ID, "labels",
            ),
        ),
    )
