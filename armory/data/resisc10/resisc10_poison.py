"""resisc10_poison dataset."""

import os

import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Subset of NWPU-RESISC45 image dataset with 10 classes, each class containing 700 images,
each image is 256 pixels by 256 pixels.
Train has 500 images, validation has 100 images, and test has 100 images.
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

_DL_URL = "https://armory-public-data.s3.us-east-2.amazonaws.com/resisc45/resisc_poison_256x256.tar.gz"
_LABELS = [
    "airplane",
    "airport",
    "harbor",
    "industrial_area",
    "railway",
    "railway_station",
    "runway",
    "ship",
    "storage_tank",
    "thermal_power_station",
]


class Resisc10Poison(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for resisc10_poison dataset."""

    VERSION = tfds.core.Version("1.1.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "Update dataset from 64x64 images to 256x256 images",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(256, 256, 3)),
                    "label": tfds.features.ClassLabel(names=_LABELS),
                }
            ),
            supervised_keys=("image", "label"),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(_DL_URL)
        splits = []
        for subdir, split in zip(
            ["train", "val", "test"],
            [tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST],
        ):
            splits.append(
                tfds.core.SplitGenerator(
                    name=split,
                    gen_kwargs={"path": os.path.join(path, "data_original", subdir)},
                )
            )
        return splits

    def _generate_examples(self, path):
        """Yields examples."""
        for label in _LABELS:
            for filename in tf.io.gfile.glob(os.path.join(path, label, "*.jpg")):
                example = {
                    "image": filename,
                    "label": label,
                }
                yield filename, example
