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

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Tensor(shape=[224, 224, 3], dtype=tf.uint8),
                    "label": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            supervised_keys=("image", "label"),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(_URL)
        splits = [
            tfds.core.SplitGenerator(name=x, gen_kwargs={"path": path, "split": x})
            for x in ("adversarial", "clean")
        ]
        return splits

    def _generate_examples(self, path, split):
        """Yields examples."""
        if split == "adversarial":
            key = "adv-image"
        elif split == "clean":
            key = "clean-image"
        else:
            raise ValueError(f"split {split} not in ('adversarial', 'clean')")

        def _parse(serialized_example, key):
            ds_features = {
                "height": tf.io.FixedLenFeature([], tf.int64),
                "width": tf.io.FixedLenFeature([], tf.int64),
                "label": tf.io.FixedLenFeature([], tf.int64),
                "adv-image": tf.io.FixedLenFeature([], tf.string),
                "clean-image": tf.io.FixedLenFeature([], tf.string),
            }
            example = tf.io.parse_single_example(serialized_example, ds_features)

            img = tf.io.decode_raw(example[key], tf.float32)
            # float values are integers in [0.0, 255.0] for clean and adversarial
            img = tf.cast(img, tf.uint8)
            img = tf.reshape(img, (example["height"], example["width"], 3))
            return img, example["label"]

        ds = tf.data.TFRecordDataset(filenames=[path])
        ds = ds.map(lambda x: _parse(x, key))
        ds = ds.batch(1)
        default_graph = tf.compat.v1.keras.backend.get_session().graph
        ds = tfds.as_numpy(ds, graph=default_graph)
        for i, (img, label) in enumerate(ds):
            yield str(i), {"image": img[0], "label": label[0]}
