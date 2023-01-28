"""apricot_dev dataset."""

import tensorflow_datasets as tfds
from . import apricot_dev


class ApricotDevTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for apricot_dev dataset."""

    # TODO(apricot_dev):
    DATASET_CLASS = apricot_dev.ApricotDev
    SPLITS = {
        "train": 3,  # Number of fake train example
        "test": 1,  # Number of fake test example
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == "__main__":
    tfds.testing.test_main()
