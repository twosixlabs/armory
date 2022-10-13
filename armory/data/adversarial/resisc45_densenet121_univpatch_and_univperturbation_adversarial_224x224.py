"""
TensorFlow Dataset for resisc45 attacked by Adversarial Patch with adv/clean splits
"""
from __future__ import absolute_import, division, print_function

import os

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
NWPU-RESISC45 adversarial image dataset for DenseNet121 using Adversarial Patch and Universal Perturbation
DenseNet121 training on benign images:
    DenseNet121 with no_top settings plus GlobalAveragePoolingLayer2D and Dense layer (45 classes), batch_size 32, initial learning rate 0.001, optimizer configuration Adam, validation accuracy 93%

Dataset contains five images from each class, totaling 225 images. All images are of the size (224,224,3). For each image, a clean version, an adversarial universal patch image, and an adversarial universal perturbation image are included.
"""

_CITATION = """
@article{cheng2017remote,
  title={Remote sensing image scene classification: Benchmark and state of the art},
  author={Cheng, Gong and Han, Junwei and Lu, Xiaoqiang},
  journal={Proceedings of the IEEE},
  volume={105},
  number={10},
  pages={1865--1883},
  year={2017},
  publisher={IEEE}
}
"""

_URL = ""
_DL_URL = (
    "https://armory-public-data.s3.us-east-2.amazonaws.com/adversarial-datasets/"
    "resisc45_densenet121_univpatch_and_univperturbation_adversarial_224x224_1.0.1.tar.gz"
)
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
_TARGET_CLASS = 21


class Resisc45Densenet121UnivpatchAndUnivperturbationAdversarial224x224(
    tfds.core.GeneratorBasedBuilder
):

    VERSION = tfds.core.Version("1.0.2")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "images": {
                        "clean": tfds.features.Image(
                            shape=[224, 224, 3], dtype=tf.uint8, encoding_format="png"
                        ),
                        "adversarial_univperturbation": tfds.features.Image(
                            shape=[224, 224, 3], dtype=tf.uint8, encoding_format="png"
                        ),
                        "adversarial_univpatch": tfds.features.Image(
                            shape=[224, 224, 3], dtype=tf.uint8, encoding_format="png"
                        ),
                    },
                    "labels": {
                        "clean": tfds.features.ClassLabel(names=_LABELS),
                        "adversarial_univperturbation": tfds.features.ClassLabel(
                            names=_LABELS
                        ),
                        "adversarial_univpatch": tfds.features.ClassLabel(
                            names=_LABELS
                        ),
                    },
                    "imagename": tfds.features.Text(),
                }
            ),
            supervised_keys=("images", "labels"),
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
            os.path.join(data_dir_path, root_dir, split_dirs[0])
        )
        labels.sort()
        for label in labels:
            image_clean = tf.io.gfile.glob(
                os.path.join(
                    data_dir_path,
                    root_dir,
                    split_dirs[0],
                    label,
                    "*.png",
                )
            )
            image_clean.sort()
            adv_univpatch = tf.io.gfile.glob(
                os.path.join(
                    data_dir_path,
                    root_dir,
                    split_dirs[1],
                    label,
                    "*.png",
                )
            )
            adv_univpatch.sort()
            adv_univperturbation = tf.io.gfile.glob(
                os.path.join(
                    data_dir_path,
                    root_dir,
                    split_dirs[2],
                    label,
                    "*.png",
                )
            )
            adv_univperturbation.sort()

            for (
                clean_img_path,
                adv_univpatch_img_path,
                adv_univperturbation_img_path,
            ) in zip(image_clean, adv_univpatch, adv_univperturbation):
                imagename = clean_img_path.split("/")[-1]
                example = {
                    "images": {
                        "clean": clean_img_path,
                        "adversarial_univpatch": adv_univpatch_img_path,
                        "adversarial_univperturbation": adv_univperturbation_img_path,
                    },
                    "labels": {
                        "clean": label,
                        "adversarial_univperturbation": label,  # untargeted, label not used
                        "adversarial_univpatch": labels[_TARGET_CLASS],  # targeted
                    },
                    "imagename": imagename,
                }
                yield imagename, example
