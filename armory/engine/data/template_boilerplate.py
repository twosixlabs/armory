fn_template = """
def {name}(
    split: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = {name}_canonical_preprocessing,
    label_preprocessing_fn: Callable = None,
    as_supervised: bool = True,
    supervised_xy_keys=None,  # May need to update value
    download_and_prepare_kwargs=None,  # May need to update value
    variable_y=False,  # May need to update value
    lambda_map: Callable = None,  # May need to update value
    fit_preprocessing_fn: Callable = None,  # May need to update value
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = True,
    **kwargs,
) -> ArmoryDataGenerator:
    preprocessing_fn = preprocessing_chain(preprocessing_fn, fit_preprocessing_fn)

    return _generator_from_tfds(
        "{ds_name}",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        as_supervised=as_supervised,
        supervised_xy_keys=supervised_xy_keys,
        download_and_prepare_kwargs=download_and_prepare_kwargs,
        variable_length=bool(batch_size > 1),
        variable_y=variable_y,
        lambda_map=lambda_map,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
        context={name}_context,
        **kwargs,
    )
"""
