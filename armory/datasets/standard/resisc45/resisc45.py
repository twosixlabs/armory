"""
TensorFlow Dataset for resisc45 with train/validate/test splits
"""

import os

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@article{Cheng_2017,
   title={Remote Sensing Image Scene Classification: Benchmark and State of the Art},
   volume={105},
   ISSN={1558-2256},
   url={http://dx.doi.org/10.1109/JPROC.2017.2675998},
   DOI={10.1109/jproc.2017.2675998},
   number={10},
   journal={Proceedings of the IEEE},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Cheng, Gong and Han, Junwei and Lu, Xiaoqiang},
   year={2017},
   month={Oct},
   pages={1865-1883}
}"""

_DESCRIPTION = """\
RESISC45 dataset is a publicly available benchmark for Remote Sensing Image
Scene Classification (RESISC), created by Northwestern Polytechnical University
(NWPU). This dataset contains 31,500 images, covering 45 scene classes with 700
images in each class. This is a variation of that dataset with 500/100/100
train/validate/test splits for each class.
"""

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

_HOMEPAGE = "http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html"

_URL = "https://armory-public-data.s3.us-east-2.amazonaws.com/resisc45/NWPU-RESISC45.tar.gz"


class Resisc45(tfds.core.GeneratorBasedBuilder):
    """NWPU Remote Sensing Image Scene Classification (RESISC) Dataset."""

    VERSION = tfds.core.Version("3.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=[256, 256, 3]),
                    "label": tfds.features.ClassLabel(names=_LABELS),
                    "filename": tfds.features.Text(),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(_URL)
        return {
            split: self._generate_examples(path, split)
            for split in ["train", "validation", "test"]
        }

    def _generate_examples(self, path, split):
        """Yields examples."""
        for label in tf.io.gfile.listdir(f"{path}/NWPU-RESISC45"):
            for idx, filename in enumerate(
                tf.io.gfile.glob(f"{path}/NWPU-RESISC45/{label}/*.jpg")
            ):
                example = {
                    "image": filename,
                    "label": label,
                    "filename": os.path.basename(filename),
                }
                if idx <= 500:
                    split_idx = "train"
                elif idx <= 600:
                    split_idx = "validation"
                else:
                    split_idx = "test"
                if split_idx == split:
                    yield filename, example
