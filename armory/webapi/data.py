"""
API queries to download and use common datasets.
"""
import tensorflow_datasets as tfds


def mnist_data() -> (dict, dict):
    """
    returns:
        Tuple of dictionaries containing numpy arrays. Keys are {`image`, `label`}
    """

    # TODO: Return generators instead of all data in memory
    train_ds = tfds.load(
        "mnist", split=tfds.Split.TRAIN, batch_size=-1, data_dir="datasets/"
    )
    test_ds = tfds.load(
        "mnist", split=tfds.Split.TEST, batch_size=-1, data_dir="datasets/"
    )

    return (
        tfds.as_numpy(train_ds),
        tfds.as_numpy(test_ds),
    )


SUPPORTED_DATASETS = {"mnist": mnist_data}
