fn_template = """
def {name}(
    split: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = {name}_canonical_preprocessing,
    fit_preprocessing_fn: Callable = None,
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
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
        context={name}_context,
        **kwargs,
    )
"""
