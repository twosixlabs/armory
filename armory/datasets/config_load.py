"""
Temporary file for testing loading from config without modifying armory.utils
"""

import copy

from armory.datasets import load, preprocessing, generator


def actual_loader(
    name,
    version=None,
    split="test",
    shuffle_files=False,
    framework="numpy",
    epochs=1,
    drop_remainder=False,
    num_batches=None,
    batch_size=1,
    shuffle_elements=False,
):
    info, ds_dict = load.load(name, version=version, shuffle_files=shuffle_files)
    preprocessor = preprocessing.get(name, version=version)
    return generator.ArmoryDataGenerator(
        info,
        ds_dict,
        split=split,
        batch_size=batch_size,
        framework=framework,
        epochs=epochs,
        drop_remainder=drop_remainder,
        element_filter=None,
        element_map=preprocessor,
        shuffle_elements=shuffle_elements,
    )


def load_dataset(dataset_config, *args, check_run=False, **kwargs):
    """
    Designed to be a drop-in replacement for armory.utils.config_loading.load_dataset

    NOTE: very ugly
    """
    dataset_config = copy.deepcopy(dataset_config)
    module = dataset_config.pop("module")
    if module == "armory.data.datasets":
        pass
    elif module == "armory.data.adversarial_datasets":
        pass  # TODO: check
    else:
        raise NotImplementedError

    name = dataset_config.pop("name")
    if ":" in name:
        name, version = name.split(":")
    else:
        version = None

    for k in "eval_split", "train_split":
        kwargs.pop(k, None)

    kwargs.update(dataset_config)
    if check_run:
        kwargs["epochs"] = 1
        kwargs["num_batches"] = 1
    if kwargs.get("shuffle_files", False):
        kwargs["shuffle_elements"] = True

    armory_data_generator = actual_loader(name, version=version, **kwargs)
    return wrap_generator(armory_data_generator)


def wrap_generator(armory_data_generator):
    from armory.datasets import art_wrapper

    return art_wrapper.WrappedDataGenerator(armory_data_generator)


def hotpatch():
    """
    Temp hot patch for armory.utils.config_loading
    """

    from armory.utils import config_loading

    config_loading.load_dataset = load_dataset
