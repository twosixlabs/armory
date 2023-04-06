"""apricot_test dataset."""

import collections
import json
import os

import tensorflow.compat.v1 as tf
import tensorflow_datasets.public_api as tfds

from armory.data.adversarial.apricot_metadata import APRICOT_MODELS, APRICOT_PATCHES

_CITATION = """
@misc{braunegg2020apricot,
      title={APRICOT: A Dataset of Physical Adversarial Attacks on Object Detection},
      author={A. Braunegg and Amartya Chakraborty and Michael Krumdick and Nicole Lape and Sara Leary and Keith Manville and Elizabeth Merkhofer and Laura Strickhart and Matthew Walmer},
      year={2020},
      eprint={1912.08166},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

_DESCRIPTION = """
The APRICOT Dataset
============================================================================
A collection of Adversarial Patches Rearranged In COnText
Version: 1.1

============================================================================

ANNOTATION FORMAT
-----------------
Annotations for APRICOT are provided in a JSON format similar to COCO, but
with extra metadata for the adversarial patches. You can find information
about the COCO format here: https://cocodataset.org/#format-data. The
APRICOT JSON files are compatible with the COCO API:
https://github.com/cocodataset/cocoapi. Object and patch locations are
provided in bounding box format only.

The fields in the APRICOT JSON format are as follows:

- categories: A list of the COCO categories. The ‘adversarial patch’
  category has been added as id 12 since that id was not used by COCO.

- patches: A list of the 60 patches used in APRICOT. Each patch gets a
  unique name and id and some additional information:
  - adv_target: The category_id of the target category
  - adv_model: The id of the detector used to generate the patch
  - is_square and is_circle: Two booleans that tell the shape of the patch

- models: A list of the models used to generate the adversarial patches.
  They have unique ids referenced by the list of patches, the name of the
  model, classifier backbone, and a link to the model from the tensorflow
  object detection API.

- info: High level information about the dataset

- annotations: A list of bounding box annotations for the objects and
  patches in APRICOT. Similar to COCO annotations, but with the following
  additional information:
  - angle: The viewing angle of the patch. 0 = head on, 1 = slightly off
    angle, 2 = severe angle. Each patch has annotations from 3 people.
  - is_warped: The patches were printed on heavy cardstock, but in some
    images you will see the patches are warped from the paper bending. We
    annotated this so we could later measure the effect of warped patches.

- images: A list of the images in APRICOT, following the COCO format.

============================================================================

DATASET PARTITIONS
------------------
The APRICOT dataset is divided into two partitions, a development partition
(dev) for validation and a testing partition (test) for reporting results.
For dataset herein, only the test partition is provided.

These files are provided as a convenience to help with different types of
experiments. For example, experiments which aim to detect patches directly
should probably be run with apricot_*_patch_annotations.json, while
experiments designed to measure COCO performance before and after a
defensive mechanism is added should use apricot_*_coco_annotations.json.

============================================================================

LEGAL
-----
The APRICOT Dataset is distributed under an Apache License Version 2.0.

Copyright 2019 The MITRE Corporation. All rights reserved. Approved for
public release; distribution unlimited.  Case #19-3440.
"""

_URL = "https://arxiv.org/abs/1912.08166"

_URLS = "https://armory-public-data.s3.us-east-2.amazonaws.com/adversarial-datasets/apricot_test.tar.gz"


class ApricotTest(tfds.core.GeneratorBasedBuilder):
    """APRICOT Test dataset."""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        features = {
            "image": tfds.features.Image(encoding_format="jpeg"),
            "images": tfds.features.Sequence(
                tfds.features.FeaturesDict(
                    {
                        "file_name": tfds.features.Text(),
                        "height": tf.int64,
                        "width": tf.int64,
                        "id": tf.int64,
                    }
                )
            ),
            "categories": tfds.features.Sequence(
                tfds.features.FeaturesDict(
                    {
                        "id": tf.int64,
                        "name": tfds.features.Text(),
                        "supercategory": tfds.features.Text(),
                    }
                )
            ),
            "models": tfds.features.Sequence(
                tfds.features.FeaturesDict(
                    {
                        "classifier": tfds.features.Text(),
                        "id": tf.int64,
                        "model": tfds.features.Text(),
                        "url": tfds.features.Text(),
                    }
                )
            ),
            "patches": tfds.features.Sequence(
                tfds.features.FeaturesDict(
                    {
                        "adv_model": tf.int64,
                        "adv_target": tf.int64,
                        "id": tf.int64,
                        "is_circle": tf.bool,
                        "is_square": tf.bool,
                        "name": tfds.features.Text(),
                    }
                )
            ),
            "objects": tfds.features.Sequence(
                {
                    "id": tf.int64,
                    "image_id": tf.int64,
                    "area": tf.int64,  # un-normalized area
                    "boxes": tfds.features.BBoxFeature(),  # normalized bounding box [ymin, xmin, ymax, xmax]
                    "labels": tfds.features.ClassLabel(num_classes=91),
                    "is_crowd": tf.bool,
                    "is_warped": tf.bool,
                    "angle": tfds.features.Tensor(shape=(3,), dtype=tf.int64),
                    "patch_id": tf.int64,
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
        return [
            tfds.core.SplitGenerator(
                name=split,
                gen_kwargs={"path": os.path.join(paths, "APRICOT"), "model": split},
            )
            for split in ["retinanet", "frcnn", "ssd"]
        ]

    def _generate_examples(self, path, model):
        """yield examples"""
        annotation_path = os.path.join(
            path, "Annotations/apricot_test_all_annotations.json"
        )
        apricot = ApricotAnnotation(annotation_path)
        images = apricot.images()

        # iterate over all images
        for image_info in sorted(images, key=lambda x: x["id"]):
            annotations = apricot.get_annotations(image_info["id"])

            def build_bbox(x, y, width, height):
                return tfds.features.BBox(
                    ymin=y / image_info["height"],
                    xmin=x / image_info["width"],
                    ymax=(y + height) / image_info["height"],
                    xmax=(x + width) / image_info["width"],
                )

            example = {
                "image": os.path.join(path, "Images/test", image_info["file_name"]),
                "images": apricot.images(),
                "categories": apricot.categories(),
                "models": apricot.models(),
                "patches": apricot.patches(),
                "objects": [
                    {
                        "id": anno["id"],
                        "image_id": anno["image_id"],
                        "area": anno["area"],
                        "boxes": build_bbox(*anno["bbox"]),
                        "labels": anno["category_id"],
                        "is_crowd": bool(anno["iscrowd"]),
                        "is_warped": anno.get("is_warped", False),
                        "angle": anno.get("angle", [-1, -1, -1]),
                        "patch_id": anno.get("patch_id", -1),
                    }
                    for anno in annotations
                ],
            }

            patch_id = [
                i["patch_id"] for i in example["objects"] if i["patch_id"] != -1
            ][0]
            adv_model_id = APRICOT_PATCHES[patch_id]["adv_model"]
            model_name = [
                i["model"] for i in APRICOT_MODELS if i["id"] == adv_model_id
            ][0]
            if model_name == model:
                yield image_info["id"], example


class ApricotAnnotation(object):
    """Apricot annotation helper class."""

    def __init__(self, annotation_path):
        with tf.io.gfile.GFile(annotation_path) as f:
            data = json.load(f)
        self._data = data

        img_id2annotations = collections.defaultdict(list)
        for a in self._data["annotations"]:
            img_id2annotations[a["image_id"]].append(a)
        self._img_id2annotations = {
            k: list(sorted(v, key=lambda a: a["id"]))
            for k, v in img_id2annotations.items()
        }

    def categories(self):
        """Return the category dicts, as sorted in the file."""
        return self._data["categories"]

    def images(self):
        """Return the image dicts, as sorted in the file."""
        return self._data["images"]

    def models(self):
        """Return the model dicts, as sorted in the file."""
        return self._data["models"]

    def patches(self):
        """Return the patch dicts, as sorted in the file."""
        return self._data["patches"]

    def get_annotations(self, img_id):
        """Return all annotations associated with the image id string."""
        return self._img_id2annotations.get(img_id, [])
