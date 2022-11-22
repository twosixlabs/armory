"""
Temporary file for testing loading from config without modifying armory.utils
"""

from armory.datasets import load, preprocessing, generator, filtering


def load_dataset(
    name=None,
    version=None,
    batch_size=1,
    num_batches=None,
    epochs=1,
    split="test",
    framework="numpy",
    preprocessor_name=None,
    preprocessor_kwargs=None,
    shuffle_files=False,
    label_key="label",  # TODO: make this smarter or more flexible
    index=None,
    class_ids=None,
    drop_remainder=False,
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
        if name in preprocessing.list_registered():
            preprocessor = preprocessing.get(name)
    else:
        preprocessor = preprocessing.get(preprocessor_name)

    if preprocessor_kwargs is not None:
        preprocessing_fn = lambda x: preprocessor(x, **preprocessor_kwargs)
    else:
        preprocessing_fn = preprocessor

    armory_data_generator = generator.ArmoryDataGenerator(
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
        element_map=preprocessing_fn,
        shuffle_elements=shuffle_files,
    )
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
