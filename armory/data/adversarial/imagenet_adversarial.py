"""
TensorFlow Dataset for resisc45 with train/validate/test splits
"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
ILSVRC12 adversarial image dataset for ResNet50

ProjectedGradientDescent
    Iterations = 10
    Max perturbation epsilon = 8
    Attack step size = 2
    Targeted = True
"""

_URL = "https://armory-public-data.s3.us-east-2.amazonaws.com/imagenet-adv/ILSVRC12_ResNet50_PGD_adversarial_dataset_v1.0.tfrecords"


class ImagenetAdversarial(tfds.core.GeneratorBasedBuilder):
    """ILSVRC12_ResNet50_PGD_adversarial_dataset_v1.0.tfrecords"""

    VERSION = tfds.core.Version("1.1.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "images": {
                        "clean": tfds.features.Tensor(
                            shape=[224, 224, 3], dtype=tf.uint8
                        ),
                        "adversarial": tfds.features.Tensor(
                            shape=[224, 224, 3], dtype=tf.uint8
                        ),
                    },
                    "label": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            supervised_keys=("images", "label"),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(_URL)
        return [tfds.core.SplitGenerator(name="adversarial", gen_kwargs={"path": path})]

    def _generate_examples(self, path):
        """Yields examples."""

        clean_key = "clean"
        adversarial_key = "adversarial"

        def _parse(serialized_example):
            ds_features = {
                "height": tf.io.FixedLenFeature([], tf.int64),
                "width": tf.io.FixedLenFeature([], tf.int64),
                "label": tf.io.FixedLenFeature([], tf.int64),
                "adv-image": tf.io.FixedLenFeature([], tf.string),
                "clean-image": tf.io.FixedLenFeature([], tf.string),
            }
            example = tf.io.parse_single_example(serialized_example, ds_features)

            img_clean = tf.io.decode_raw(example["clean-image"], tf.float32)
            img_adv = tf.io.decode_raw(example["adv-image"], tf.float32)
            # float values are integers in [0.0, 255.0] for clean and adversarial
            img_clean = tf.cast(img_clean, tf.uint8)
            img_clean = tf.reshape(img_clean, (example["height"], example["width"], 3))
            img_adv = tf.cast(img_adv, tf.uint8)
            img_adv = tf.reshape(img_adv, (example["height"], example["width"], 3))
            return {clean_key: img_clean, adversarial_key: img_adv}, example["label"]

        ds = tf.data.TFRecordDataset(filenames=[path])
        ds = ds.map(lambda x: _parse(x))

        for i, (img, label) in enumerate(tfds.as_numpy(ds)):
            yield str(i), {
                "images": img,
                "label": label,
            }
