"""
TensorFlow poison dataset for GTSRB
"""

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
GTSRB poison sample dataset for micronNet, poisoning class 1 with custom bullet-hole pattern
images are resampled to 48x48

"""

_DL_URL = "https://armory-public-data.s3.us-east-2.amazonaws.com/adversarial-datasets/gtsrb_poisoned_images_entire_class.npy"
_TEST_URL = "https://armory-public-data.s3.us-east-2.amazonaws.com/adversarial-datasets/gtsrb_poisoned_test_images.npy"


class GtsrbBhPoisonMicronnet(tfds.core.GeneratorBasedBuilder):
    """GTSRB_bh_poison_micronnet"""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Tensor(shape=(48, 48, 3), dtype=tf.float32),
                    "label": tfds.features.ClassLabel(num_classes=43),
                }
            ),
            supervised_keys=("image", "label"),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_path = dl_manager.download(_DL_URL)
        test_path = dl_manager.download(_TEST_URL)
        return [
            tfds.core.SplitGenerator(
                name="poison",
                gen_kwargs={
                    "data_dir_path": dl_path,
                },
            ),
            tfds.core.SplitGenerator(
                name="poison_test",
                gen_kwargs={
                    "data_dir_path": test_path,
                },
            ),
        ]

    def _generate_examples(self, data_dir_path):
        """Yields examples."""

        with tf.io.gfile.GFile(data_dir_path, "rb") as fp:
            images = np.load(fp)
        labels = np.full(images.shape[0], 2)
        for i, img in enumerate(images):
            yield i, {"image": img, "label": labels[i]}
