import os
import unittest

from armory import paths
from armory.data.utils import download_file_from_s3


class ModelWeightsTest(unittest.TestCase):
    def test_download(self):
        saved_model_dir = paths.host().saved_model_dir
        weights_file = "resnet50_weights_tf_dim_ordering_tf_kernels.h5"

        download_file_from_s3(
            "armory-public-data",
            f"model-weights/{weights_file}",
            f"{saved_model_dir}/{weights_file}",
        )
        assert os.path.isfile(os.path.join(saved_model_dir, weights_file))
