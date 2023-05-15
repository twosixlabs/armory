"""MS Coco Poisoning Dataset."""

from __future__ import annotations

import collections
import json
import os

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

_CITATION = """\
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{\'{a}}r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  archivePrefix = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/LinMBHPRDZ14},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """COCO is a large-scale object detection, segmentation, and
captioning dataset.

Note:
 * Coco 2017 is used for better train/val splits
 * Only a subset of the 80 classes are used for the poisoning dataset
"""

_CONFIG_DESCRIPTION = """
This version contains images, bounding boxes and labels for the 2017 version.
"""
# Keep these category:id for the poisoning dataset
# airplane:5
# bus:6
# train:7
#
# Train
# Total images: 10349
# Total class instances: [(5, 5135), (6, 6069), (7, 4571)]
# Val
# Total images: 434
# Total class instances: [(5, 143), (6, 285), (7, 190)]
_KEEP_CLASSES_ID = [5, 6, 7]

Split = collections.namedtuple("Split", ["name", "images", "annotations"])


class CocoConfig(tfds.core.BuilderConfig):
    """BuilderConfig for CocoConfig."""

    def __init__(self, splits=None, **kwargs):
        super(CocoConfig, self).__init__(version=tfds.core.Version("1.0.0"), **kwargs)
        self.splits = splits


class Minicoco(tfds.core.GeneratorBasedBuilder):
    """
    MS Coco poisoning dataset.
    Derived from https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/coco.py
    """

    BUILDER_CONFIGS = [
        CocoConfig(
            name="2017",
            description=_CONFIG_DESCRIPTION.format(year=2017),
            splits=[
                Split(
                    name=tfds.Split.TRAIN,
                    images="train2017",
                    annotations="annotations_trainval2017",
                ),
                Split(
                    name=tfds.Split.VALIDATION,
                    images="val2017",
                    annotations="annotations_trainval2017",
                ),
            ],
        ),
    ]

    def _info(self):
        features = {
            # Images can have variable shape
            "image": tfds.features.Image(encoding_format="jpeg"),
            "image/filename": tfds.features.Text(),
            "image/id": tf.int64,
            "objects": tfds.features.Sequence(
                {
                    "id": np.int64,  # Coco has unique id for each annotation.
                    "area": np.int64,
                    "bbox": tfds.features.BBoxFeature(),
                    # Coco has 80 classes but only a small subset appear in the dataset
                    "label": tfds.features.ClassLabel(
                        num_classes=len(_KEEP_CLASSES_ID)
                    ),
                    "is_crowd": np.bool_,
                }
            ),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="http://cocodataset.org/#home",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        # Merge urls from all splits together
        urls = {}
        for split in self.builder_config.splits:
            urls["{}_images".format(split.name)] = "zips/{}.zip".format(split.images)
            urls["{}_annotations".format(split.name)] = "annotations/{}.zip".format(
                split.annotations
            )

        # DownloadManager memoize the url, so duplicate urls will only be downloaded
        # once.
        root_url = "http://images.cocodataset.org/"
        extracted_paths = dl_manager.download_and_extract(
            {key: root_url + url for key, url in urls.items()}
        )

        splits = []
        for split in self.builder_config.splits:
            image_dir = extracted_paths["{}_images".format(split.name)]
            annotations_dir = extracted_paths["{}_annotations".format(split.name)]
            splits.append(
                tfds.core.SplitGenerator(
                    name=split.name,
                    gen_kwargs=dict(
                        image_dir=image_dir,
                        annotation_dir=annotations_dir,
                        split_name=split.images,
                    ),
                )
            )
        return splits

    def _generate_examples(self, image_dir, annotation_dir, split_name):
        """Generate examples as dicts.

        Args:
          image_dir: `str`, directory containing the images
          annotation_dir: `str`, directory containing annotations
          split_name: `str`, <split_name><year> (ex: train2014, val2017)

        Yields:
          example key and data
        """
        instance_filename = "instances_{}.json"

        # Load the annotations (label names, images metadata,...)
        instance_path = os.path.join(
            annotation_dir,
            "annotations",
            instance_filename.format(split_name),
        )
        coco_annotation = CocoAnnotationBBoxes(instance_path)
        # Each category is a dict:
        # {
        #    'id': 51,  # From 1-91, some entry missing
        #    'name': 'bowl',
        #    'supercategory': 'kitchen',
        # }
        categories = coco_annotation.categories
        # Each image is a dict:
        # {
        #     'id': 262145,
        #     'file_name': 'COCO_train2017_000000262145.jpg'
        #     'flickr_url': 'http://farm8.staticflickr.com/7187/xyz.jpg',
        #     'coco_url': 'http://images.cocodataset.org/train2017/xyz.jpg',
        #     'license': 2,
        #     'date_captured': '2013-11-20 02:07:55',
        #     'height': 427,
        #     'width': 640,
        # }
        images = coco_annotation.images

        objects_key = "objects"
        self.info.features[objects_key]["label"].names = [c["name"] for c in categories]

        categories_id2name = {c["id"]: c["name"] for c in categories}

        # Iterate over all images
        for image_info in sorted(images, key=lambda x: x["id"]):
            # Each instance annotation is a dict:
            # {
            #     'iscrowd': 0,
            #     'bbox': [116.95, 305.86, 285.3, 266.03],
            #     'image_id': 480023,
            #     'segmentation': [[312.29, 562.89, 402.25, ...]],
            #     'category_id': 58,
            #     'area': 54652.9556,
            #     'id': 86,
            # }
            instances = coco_annotation.get_annotations(img_id=image_info["id"])

            if not instances:
                continue

            def build_bbox(x, y, width, height):
                # pylint: disable=cell-var-from-loop
                # build_bbox is only used within the loop so it is ok to use image_info
                return tfds.features.BBox(
                    ymin=y / image_info["height"],
                    xmin=x / image_info["width"],
                    ymax=(y + height) / image_info["height"],
                    xmax=(x + width) / image_info["width"],
                )
                # pylint: enable=cell-var-from-loop

            example = {
                "image": os.path.join(image_dir, split_name, image_info["file_name"]),
                "image/filename": image_info["file_name"],
                "image/id": image_info["id"],
                objects_key: [
                    {  # pylint: disable=g-complex-comprehension
                        "id": instance["id"],
                        "area": instance["area"],
                        "bbox": build_bbox(*instance["bbox"]),
                        "label": categories_id2name[instance["category_id"]],
                        "is_crowd": bool(instance["iscrowd"]),
                    }
                    for instance in instances
                ],
            }

            yield image_info["file_name"], example


class CocoAnnotation(object):
    """Coco annotation helper class."""

    def __init__(self, annotation_path):
        with tf.io.gfile.GFile(annotation_path) as f:
            data = json.load(f)
        self._data = data

    @property
    def categories(self):
        """Return the category dicts, as sorted in the file."""
        categories = []
        for cat in self._data["categories"]:
            if cat["id"] in _KEEP_CLASSES_ID:
                categories.append(cat)
        return categories

    @property
    def images(self):
        """Return the image dicts, as sorted in the file."""
        return self._data["images"]

    def get_annotations(self, img_id):
        """Return all annotations associated with the image id string."""
        raise NotImplementedError  # AnotationType.NONE don't have annotations


class CocoAnnotationBBoxes(CocoAnnotation):
    """Coco annotation helper class."""

    def __init__(self, annotation_path):
        super(CocoAnnotationBBoxes, self).__init__(annotation_path)

        img_id2annotations = collections.defaultdict(list)
        for a in self._data["annotations"]:
            if a["category_id"] in _KEEP_CLASSES_ID:
                img_id2annotations[a["image_id"]].append(a)
        self._img_id2annotations = {
            k: list(sorted(v, key=lambda a: a["id"]))
            for k, v in img_id2annotations.items()
        }

    def get_annotations(self, img_id):
        """Return all annotations associated with the image id string."""
        # Some images don't have any annotations. Return empty list instead.
        return self._img_id2annotations.get(img_id, [])
