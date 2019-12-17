"""

"""

import tensorflow_datasets as tfds


def mnist_data():
    train_ds = tfds.load("mnist", split=tfds.Split.TRAIN, batch_size=-1)
    test_ds = tfds.load("mnist", split=tfds.Split.TEST, batch_size=-1)

    return (
        tfds.as_numpy(train_ds),
        tfds.as_numpy(test_ds),
    )


SUPPORTED_DATASETS = {"mnist": mnist_data}
