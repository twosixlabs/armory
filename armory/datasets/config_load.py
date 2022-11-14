"""
Temporary file for testing loading from config without modifying armory.utils
"""

import copy

from armory.datasets import load, preprocessing, generator, filtering


def actual_loader(
    name=None,
    version=None,
    shuffle_files=False,
    preprocessor_name="DEFAULT",
    split="test",
    framework="numpy",
    epochs=1,
    drop_remainder=False,
    num_batches=None,
    batch_size=1,
    shuffle_elements=False,
    label_key="label",  # TODO: make this smarter or more flexible
    class_ids=None,
    index=None,
):
    # All are keyword elements by design
    if name is None:
        raise ValueError("name must be specified, not None")
    info, ds_dict = load.load(name, version=version, shuffle_files=shuffle_files)

    if class_ids is None:
        element_filter = None
    else:
        if isinstance(class_ids, int):
            class_ids = [class_ids]
        if not isinstance(class_ids, list):
            raise ValueError(
                f"class_ids must be a list, int, or None, not {type(class_ids)}"
            )
        element_filter = filtering.get_filter_by_class(class_ids, label_key=label_key)
    if index is None:
        index_filter = None
    elif isinstance(index, list):
        index_filter = filtering.get_enum_filter_by_index(index)
    elif isinstance(index, str):
        index_filter = filtering.get_enum_filter_by_slice(index)
    else:
        raise ValueError(f"index must be a list, str, or None, not {type(index)}")

    if preprocessor_name is None:
        preprocessor = None
    elif preprocessor_name == "DEFAULT":
        # TODO: do something smart here
        if preprocessing.has(name):
            preprocessor = preprocessing.get(preprocessor_name)
        else:
            preprocessor = preprocessing.infer_from_dataset_info(info, split)

        raise NotImplementedError
    else:
        preprocessor = preprocessing.get(preprocessor_name)

    return generator.ArmoryDataGenerator(
        info,
        ds_dict,
        split=split,
        batch_size=batch_size,
        framework=framework,
        epochs=epochs,
        drop_remainder=drop_remainder,
        num_batches=num_batches,
        index_filter=index_filter,
        element_filter=element_filter,
        element_map=preprocessor,
        key_map=None,
        shuffle_elements=shuffle_elements,
    )


def load_dataset(dataset_config, *args, check_run=False, **kwargs):
    """
    Designed to be a drop-in replacement for armory.utils.config_loading.load_dataset

    NOTE: very ugly
    """

    if args:
        raise NotImplementedError("args not supported here")
    kwargs.update(dataset_config)
    module = kwargs.pop("module")
    if module == "armory.data.datasets":
        pass
    elif module == "armory.data.adversarial_datasets":
        pass  # TODO: check
    else:
        # NOTE: temporary until moving over to new datasets approach
        raise NotImplementedError

    name = kwargs.pop("name")
    if ":" in name:
        name, version = name.split(":")
    else:
        version = None
    kwargs["name"] = name
    kwargs["version"] = version

    for k in "eval_split", "train_split":
        kwargs.pop(k, None)

    if check_run:
        kwargs["epochs"] = 1
        kwargs["num_batches"] = 1
    if kwargs.get("shuffle_files", False):
        kwargs["shuffle_elements"] = True

    armory_data_generator = actual_loader(**kwargs)
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
