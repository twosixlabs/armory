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
    preprocessor_name="DEFAULT",
    preprocessor_kwargs=None,
    shuffle_files=False,
    label_key=None,
    index=None,
    class_ids=None,
    drop_remainder=False,
    key_map: dict = None,
    use_supervised_keys: bool = True,
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
        if label_key is None:
            if info.supervised_keys is None:
                raise ValueError(
                    "label_key is None and info.supervised_keys is None."
                    " What label is being filtered on?"
                )
            elif len(info.supervised_keys) != 2 or not all(
                isinstance(k, str) for k in info.supervised_keys
            ):
                raise NotImplementedError(
                    f"supervised_keys {info.supervised_keys} is not a 2-tuple of str."
                    " Please specify label_key for filtering."
                )
            _, label_key = info.supervised_keys

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
        if preprocessing.has(name):
            preprocessor = preprocessing.get(name)
        else:
            preprocessor = preprocessing.infer_from_dataset_info(info, split)
    else:
        preprocessor = preprocessing.get(preprocessor_name)

    if preprocessor is not None and preprocessor_kwargs is not None:
        preprocessing_fn = lambda x: preprocessor(  # noqa: E731
            x, **preprocessor_kwargs
        )
    else:
        preprocessing_fn = preprocessor

    shuffle_elements = shuffle_files

    armory_data_generator = generator.ArmoryDataGenerator(
        info,
        ds_dict,
        split=split,
        batch_size=batch_size,
        num_batches=num_batches,
        epochs=epochs,
        drop_remainder=drop_remainder,
        index_filter=index_filter,
        element_filter=element_filter,
        element_map=preprocessing_fn,
        shuffle_elements=shuffle_elements,
        framework=framework,
    )
    # If key_map is not None, use_supervised_keys is ignored
    if key_map is not None:
        # ignore use_supervised_keys in this case
        armory_data_generator.set_key_map(key_map)
    else:
        # error if use_supervised_keys and supervised_keys do not exist in info
        armory_data_generator.set_key_map(use_supervised_keys=use_supervised_keys)

    # Let the scenario set the desired tuple directly
    # armory_data_generator.as_tuple()  # NOTE: This will currently fail for adversarial datasets

    return armory_data_generator
