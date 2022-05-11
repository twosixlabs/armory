"""xview dataset."""

import io
import os

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@article{DBLP:journals/corr/abs-1802-07856,
  author    = {Darius Lam and
               Richard Kuzma and
               Kevin McGee and
               Samuel Dooley and
               Michael Laielli and
               Matthew Klaric and
               Yaroslav Bulatov and
               Brendan McCord},
  title     = {xView: Objects in Context in Overhead Imagery},
  journal   = {CoRR},
  volume    = {abs/1802.07856},
  year      = {2018},
  url       = {http://arxiv.org/abs/1802.07856},
  archivePrefix = {arXiv},
  eprint    = {1802.07856},
  timestamp = {Mon, 13 Aug 2018 16:49:13 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1802-07856.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """
This dataset contains chips of original high-res xView images in 200x200,
300x300, 400x400, and 500x500 sizes.

The dataset is split into ~58k training images and ~19k test images
"""

_URL = "http://xviewdataset.org/"

_URLS = "https://armory-public-data.s3.us-east-2.amazonaws.com/xview/xview.tar.gz"

"""
original dataset have 62 classes whose labels are not sequential. Map to sequential classes
original mapping:
11:Fixed-wing Aircraft
12:Small Aircraft
13:Cargo Plane
15:Helicopter
17:Passenger Vehicle
18:Small Car
19:Bus
...
"""
LABEL_MAP = {
    11: 1,  # Fixed-wing Aircraft
    12: 2,  # Small Aircraft
    13: 3,  # Cargo Plane
    15: 4,  # Helicopter
    17: 5,  # Passenger Vehicle
    18: 6,  # Small Car
    19: 7,  # Bus
    20: 8,  # Pickup Truck
    21: 9,  # Utility Truck
    23: 10,  # Truck
    24: 11,  # Cargo Truck
    25: 12,  # Truck w/Box
    26: 13,  # Truck Tractor
    27: 14,  # Trailer
    28: 15,  # Truck w/Flatbed
    29: 16,  # Truck w/Liquid
    32: 17,  # Crane Truck
    33: 18,  # Railway Vehicle
    34: 19,  # Passenger Car
    35: 20,  # Cargo Car
    36: 21,  # Flat Car
    37: 22,  # Tank Car
    38: 23,  # Locomotive
    40: 24,  # Maritime Vessel
    41: 25,  # Motorboat
    42: 26,  # Sailboat
    44: 27,  # Tugboat
    45: 28,  # Barge
    47: 29,  # Fishing Vessel
    49: 30,  # Ferry
    50: 31,  # Yacht
    51: 32,  # Container Ship
    52: 33,  # Oil Tanker
    53: 34,  # Engineering Vehicle
    54: 35,  # Tower Crane
    55: 36,  # Container Crane
    56: 37,  # Reach Stacker
    57: 38,  # Straddle Carrier
    59: 39,  # Mobile Crane
    60: 40,  # Dump Truck
    61: 41,  # Haul Truck
    62: 42,  # Scraper/Tractor
    63: 43,  # Front Loader/Bulldozer
    64: 44,  # Excavator
    65: 45,  # Cement Mixer
    66: 46,  # Ground Grader
    71: 47,  # Hut/Tent
    72: 48,  # Shed
    73: 49,  # Building
    74: 50,  # Aircraft Hanger
    75: 51,  # Unknown1
    76: 52,  # Damaged Building
    77: 53,  # Facility
    79: 54,  # Construction Site
    82: 55,  # Unknown2
    83: 56,  # Vehicle Lot
    84: 57,  # Helipad
    86: 58,  # Storage Tank
    89: 59,  # Shipping Container Lot
    91: 60,  # Shipping Container
    93: 61,  # Pylon
    94: 62,  # Tower
}


class Xview(tfds.core.GeneratorBasedBuilder):
    """xView dataset."""

    VERSION = tfds.core.Version("1.0.1")

    def _info(self):
        features = {
            "image": tfds.features.Image(encoding_format="jpeg"),
            "objects": tfds.features.Sequence(
                {
                    "id": tf.int64,
                    "image_id": tf.int64,
                    "area": tf.int64,  # un-normalized area
                    "boxes": tfds.features.BBoxFeature(),  # normalized bounding box [ymin, xmin, ymax, xmax]
                    "labels": tfds.features.ClassLabel(num_classes=63),
                    "is_crowd": tf.bool,
                }
            ),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        paths = dl_manager.download_and_extract(_URLS)
        splits = []
        for file, name in zip(
            [
                "data_multires/train/xview_train_t1.record",
                "data_multires/test/xview_test_t1.record",
            ],
            [tfds.Split.TRAIN, tfds.Split.TEST],
        ):
            splits.append(
                tfds.core.SplitGenerator(
                    name=name, gen_kwargs={"path": os.path.join(paths, file)}
                )
            )

        return splits

    def _generate_examples(self, path):
        # read original tfrecord
        tfrecord_iterator = tf.python_io.tf_record_iterator(path=path)

        # iterate and parse tfrecord into tfds shards
        obj_id = 0
        for image_id, string_record in enumerate(tfrecord_iterator):
            example = tf.train.Example()
            example.ParseFromString(string_record)

            height = example.features.feature["image/height"].int64_list.value[0]

            width = example.features.feature["image/width"].int64_list.value[0]

            img_string = example.features.feature["image/encoded"].bytes_list.value[0]

            # bounding box coordinates are already normalized
            x_min = example.features.feature["image/object/bbox/xmin"].float_list.value

            x_max = example.features.feature["image/object/bbox/xmax"].float_list.value

            y_min = example.features.feature["image/object/bbox/ymin"].float_list.value

            y_max = example.features.feature["image/object/bbox/ymax"].float_list.value

            label = example.features.feature[
                "image/object/class/label"
            ].int64_list.value
            label = list(
                map(lambda x: LABEL_MAP[x], label)
            )  # map original classes to sequential classes

            annotations = {}
            annotations["box"] = [
                tfds.features.BBox(ymin, xmin, ymax, xmax)
                for (xmin, ymin, xmax, ymax) in zip(x_min, y_min, x_max, y_max)
            ]
            annotations["label"] = np.array(label, dtype=np.int64)
            annotations["image_id"] = [image_id] * len(x_min)
            annotations["id"] = [obj_id + i for i in range(len(x_min))]
            obj_id += len(x_min)
            annotations["area"] = np.array(
                [
                    width * (xmax - xmin) * height * (ymax - ymin)
                    for (xmin, ymin, xmax, ymax) in zip(x_min, y_min, x_max, y_max)
                ],
                dtype=np.int64,
            )
            annotations["iscrowd"] = np.zeros((len(label),), dtype=np.bool_)

            example = {
                "image": io.BytesIO(img_string),
                "objects": {
                    "id": annotations["id"],
                    "image_id": annotations["image_id"],
                    "area": annotations["area"],
                    "boxes": annotations["box"],
                    "labels": annotations["label"],
                    "is_crowd": annotations["iscrowd"],
                },
            }
            yield image_id, example
