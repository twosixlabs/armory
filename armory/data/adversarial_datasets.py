"""
Adversarial datasets
"""

from typing import Callable

from armory.data import datasets
from armory.data.adversarial import (  # noqa: F401
    ucf101_mars_perturbation_and_patch_adversarial_112x112,
)


def ucf101_adversarial_112x112(
    split_type: str = "adversarial",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = False,
    framework: str = "numpy",
    clean_key: str = "clean",
    adversarial_key: str = "adversarial_perturbation",
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

    return datasets._generator_from_tfds(
        "ucf101_mars_perturbation_and_patch_adversarial112x112:1.0.0",
        split_type=split_type,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("videos", "label"),
        variable_length=False,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda z: ((z[0][clean_key], z[0][adversarial_key]), z[1]),
        # x_subset_keys=(clean_key, adversarial_key),
        # ds = ds.map(lambda z: ({k:v for (k,v) in z[0] if k in x_subset_keys}, z[1]))
    )
