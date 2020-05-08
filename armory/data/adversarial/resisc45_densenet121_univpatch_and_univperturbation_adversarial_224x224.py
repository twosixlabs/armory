"""
TensorFlow Dataset for resisc45 attacked by Adversarial Patch with adv/clean splits
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
NWPU-RESISC45 adversarial image dataset for DenseNet121 using Adversarial Patch and Universal Perturbation
DenseNet121 training on benign images: 
    architecture name, batch_size, initial learning rate, optimizer configuration, loss, metrics

AdversarialPatch
    max_iter 1000
    scale_min, scale_max 0.4
    learning_rate 2.0
    (By definition targeted)
UniversalPerturbation
    attacker 'pgd'
    pgd eps 0.2
    pgd eps_step 0.05
    max_iter 20
    eps 0.1
    delta 0.5
    norm np.inf 
    untargeted
"""

_URL = ""
_CITATION = ""
_DL_URL = "/armory/datasets/resisc45_densenet121_univpatch_and_univperturbation_adversarial_224x224.tar.gz"
_LABELS = [
    "airplane",
    "airport",
    "baseball_diamond",
    "basketball_court",
    "beach",
    "bridge",
    "chaparral",
    "church",
    "circular_farmland",
    "cloud",
    "commercial_area",
    "dense_residential",
    "desert",
    "forest",
    "freeway",
    "golf_course",
    "ground_track_field",
    "harbor",
    "industrial_area",
    "intersection",
    "island",
    "lake",
    "meadow",
    "medium_residential",
    "mobile_home_park",
    "mountain",
    "overpass",
    "palace",
    "parking_lot",
    "railway",
    "railway_station",
    "rectangular_farmland",
    "river",
    "roundabout",
    "runway",
    "sea_ice",
    "ship",
    "snowberg",
    "sparse_residential",
    "stadium",
    "storage_tank",
    "tennis_court",
    "terrace",
    "thermal_power_station",
    "wetland",
]


class Resisc45Densenet121UnivpatchAndUnivperturbationAdversarial224x224(
    tfds.core.GeneratorBasedBuilder
):

    VERSION = tfds.core.Version("1.0.0")

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
                        "adversarial_univperturbation": tfds.features.Tensor(
                            shape=[224, 224, 3], dtype=tf.uint8
                        ),
                        "adversarial_univpatch": tfds.features.Tensor(
                            shape=[224, 224, 3], dtype=tf.uint8
                        ),
                    },
                    "label": tfds.features.Tensor(shape=(), dtype=tf.int64),
                    "imagename": tfds.features.Text(),
                }
            ),
            supervised_keys=("images", "label"),
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Adversarial dataset only has TEST split"""
        dl_path = dl_manager.download_and_extract(_DL_URL)
        return [
            tfds.core.SplitGenerator(
                name="adversarial", gen_kwargs={"data_dir_path": dl_path}
            )
        ]

    def _generate_examples(self, data_dir_path):
        """Yields examples."""
        root_dir = "data"
        split_dirs = ["clean", "adversarial_univpatch", "adversarial_univperturbation"]
        labels = tf.io.gfile.listdir(
            os.path.join(
                data_dir_path, root_dir, split_dirs[0]
            )  # deal with data_dir_path in _split_generator
        )
        labels.sort()
        for label in labels:
            imagenames = tf.io.gfile.listdir(
                os.path.join(data_dir_path, root_dir, split_dirs[0], label)
            )
            imagenames.sort()
            for imagename in imagenames:
                image_clean = tf.io.gfile.glob(
                    os.path.join(
                        data_dir_path,
                        root_dir,
                        split_dirs[0],
                        label,
                        imagename,
                        "*.jpg",
                    )
                )
                image_clean.sort()
                adv_univpatch = tf.io.gfile.glob(
                    os.path.join(
                        data_dir_path,
                        root_dir,
                        split_dirs[1],
                        label,
                        imagename,
                        "*.jpg",
                    )
                )
                adv_univpatch.sort()
                adv_univperturbation = tf.io.gfile.glob(
                    os.path.join(
                        data_dir_path,
                        root_dir,
                        split_dirs[2],
                        label,
                        imagename,
                        "*.jpg",
                    )
                )
                adv_univperturbation.sort()
                example = {
                    "images": {
                        "clean": image_clean,
                        "adversarial_univpatch": adv_univpatch,
                        "adversarial_univperturbation": adv_univperturbation,
                    },
                    "label": label,
                    "imagename": imagename,
                }
                yield imagename, example
