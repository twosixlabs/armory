"""dapricot_dev dataset."""

import collections
import json
import os

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from armory.data.adversarial import pandas_proxy

_DESCRIPTION = """
LEGAL
-----
Copyright 2021 The MITRE Corporation. All rights reserved.
"""

_CITATION = """
Dataset is unpublished at this time.
"""

_URLS = "https://armory-public-data.s3.us-east-2.amazonaws.com/adversarial-datasets/dapricot_dev.tar.gz"


class DapricotDev(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dapricot_dev dataset."""

    VERSION = tfds.core.Version("1.0.1")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.0.1": "Updated to access full dev dataset",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        features = {
            # all Sequences are for [camera_1, camera_2, camera_3]
            "image": tfds.features.Sequence(
                tfds.features.Image(shape=(None, None, 3)),  # encoding_format="jpeg"),
                length=3,
            ),
            "images": tfds.features.Sequence(
                tfds.features.FeaturesDict(
                    {
                        "file_name": tfds.features.Text(),
                        "height": tf.int64,
                        "width": tf.int64,
                        "id": tf.int64,
                    }
                ),
                length=3,
            ),
            "categories": tfds.features.Sequence(
                tfds.features.Sequence(
                    tfds.features.FeaturesDict(
                        {
                            "id": tf.int64,  # {'octagon':12, 'diamond':26, 'rect':29}
                            "name": tfds.features.Text(),
                        }
                    )
                ),
                length=3,
            ),
            "objects": tfds.features.Sequence(
                tfds.features.Sequence(
                    {
                        "id": tf.int64,
                        "image_id": tf.int64,
                        "area": tf.int64,  # un-normalized area
                        "boxes": tfds.features.BBoxFeature(),  # normalized bounding box [ymin, xmin, ymax, xmax]
                        "labels": tfds.features.ClassLabel(num_classes=91),
                        "is_crowd": tf.bool,
                    }
                ),
                length=3,
            ),
            "patch_metadata": tfds.features.Sequence(
                # these data only apply to the "green screen patch" objects
                tfds.features.FeaturesDict(
                    {
                        "gs_coords": tfds.features.Sequence(
                            tfds.features.Tensor(
                                shape=[2], dtype=tf.int64
                            ),  # green screen vertices in (x,y)
                        ),
                        "cc_ground_truth": tfds.features.Tensor(
                            shape=[24, 3], dtype=tf.float32
                        ),  # colorchecker color ground truth
                        "cc_scene": tfds.features.Tensor(
                            shape=[24, 3], dtype=tf.float32
                        ),  # colorchecker colors in a scene
                        "shape": tfds.features.Text(),  # "diamond", "rect", "octagon"
                    }
                ),
                length=3,
            ),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        paths = dl_manager.download_and_extract(_URLS)
        return [
            tfds.core.SplitGenerator(
                name=patch_size,
                gen_kwargs={"path": os.path.join(paths, "dev"), "size": patch_size},
            )
            for patch_size in ["large", "medium", "small"]
        ]

    def _generate_examples(self, path, size):
        """yield examples"""

        scenes = ["01", "06", "14"]

        size_dist = {"small": "dist15", "medium": "dist10", "large": "dist5"}

        yield_id = 0
        # For each scene, read JSONs for all cameras.
        # For each camera, go through each image.
        # For each image, gets its annotations and yield relevant data
        for scene in scenes:

            annotation_path_camera_1 = os.path.join(
                path, "annotations/labels_scene_{}_camera_1.json".format(scene)
            )
            annotation_path_camera_2 = os.path.join(
                path, "annotations/labels_scene_{}_camera_2.json".format(scene)
            )
            annotation_path_camera_3 = os.path.join(
                path, "annotations/labels_scene_{}_camera_3.json".format(scene)
            )

            dapricot_camera_1 = DapricotAnnotation(annotation_path_camera_1)
            dapricot_camera_2 = DapricotAnnotation(annotation_path_camera_2)
            dapricot_camera_3 = DapricotAnnotation(annotation_path_camera_3)

            images_camera_1 = dapricot_camera_1.images()
            images_camera_2 = dapricot_camera_2.images()
            images_camera_3 = dapricot_camera_3.images()

            # sort images alphabetically so all three cameras are consistent
            images_camera_1 = sorted(
                images_camera_1, key=lambda x: x["file_name"].lower()
            )
            images_camera_2 = sorted(
                images_camera_2, key=lambda x: x["file_name"].lower()
            )
            images_camera_3 = sorted(
                images_camera_3, key=lambda x: x["file_name"].lower()
            )

            for image_camera_1, image_camera_2, image_camera_3 in zip(
                images_camera_1, images_camera_2, images_camera_3
            ):

                # verify consistency
                fname1 = image_camera_1[
                    "file_name"
                ]  # fname has format "scene_#_camera_1_<SHAPE>_<HEIGHT>_<DIST>.JPG"
                fname2 = image_camera_2["file_name"]
                fname3 = image_camera_3["file_name"]
                assert fname1 == ("_").join(
                    fname2.split("_")[:3] + ["1"] + fname2.split("_")[4:]
                ), "{} and {} are inconsistent".format(fname1, fname2)
                assert fname1 == ("_").join(
                    fname3.split("_")[:3] + ["1"] + fname3.split("_")[4:]
                ), "{} and {} are inconsistent".format(fname1, fname3)

                # get object annotations for each image
                annotations_camera_1 = dapricot_camera_1.get_annotations(
                    image_camera_1["id"]
                )
                annotations_camera_2 = dapricot_camera_2.get_annotations(
                    image_camera_2["id"]
                )
                annotations_camera_3 = dapricot_camera_3.get_annotations(
                    image_camera_3["id"]
                )

                # convert bbox to Pytorch format
                def build_bbox(x, y, width, height):
                    return tfds.features.BBox(
                        ymin=y
                        / image_camera_1[
                            "height"
                        ],  # all images are the same size, so using image_camera_1 is fine
                        xmin=x / image_camera_1["width"],
                        ymax=(y + height) / image_camera_1["height"],
                        xmax=(x + width) / image_camera_1["width"],
                    )

                # convert segmentation format of (x0,y0,x1,y1,...) to ( (x0, y0), (x1, y1), ... )
                def build_coords(segmentation):
                    xs = segmentation[::2]
                    ys = segmentation[1::2]
                    coords = [[int(round(x)), int(round(y))] for (x, y) in zip(xs, ys)]

                    return coords

                # convert green screen shape given in file name to shape expected in downstream algorithms
                def get_shape(in_shape):
                    out_shape = {"stp": "octagon", "pxg": "diamond", "spd": "rect"}
                    return out_shape[in_shape]

                # get colorchecker color box values. There are 24 color boxes, so output shape is (24, 3)
                def get_cc(ground_truth=True, scene=None, camera=None):
                    if ground_truth:
                        return pandas_proxy.read_csv_to_numpy_float32(
                            os.path.join(
                                path,
                                "annotations",
                                "xrite_passport_colors_sRGB-GMB-2005.csv",
                            ),
                            header=None,
                        )
                    else:
                        return pandas_proxy.read_csv_to_numpy_float32(
                            os.path.join(
                                path,
                                "annotations",
                                "scene_{}_camera_{}_CC_values.csv".format(
                                    scene, camera
                                ),
                            ),
                            header=None,
                        )

                example = {
                    "image": [
                        os.path.join(
                            path,
                            "scene_{}/camera_{}".format(scene, camera + 1),
                            im_cam["file_name"],
                        )
                        for camera, im_cam in enumerate(
                            [image_camera_1, image_camera_2, image_camera_3]
                        )
                    ],
                    "images": [image_camera_1, image_camera_2, image_camera_3],
                    "categories": [
                        d_cam.categories()
                        for d_cam in [
                            dapricot_camera_1,
                            dapricot_camera_2,
                            dapricot_camera_3,
                        ]
                    ],
                    "objects": [
                        [
                            {
                                "id": anno["id"],
                                "image_id": anno["image_id"],
                                "area": anno["area"],
                                "boxes": build_bbox(*anno["bbox"]),
                                "labels": anno["category_id"],
                                "is_crowd": bool(anno["iscrowd"]),
                            }
                            for anno in annos
                        ]
                        for annos in [
                            annotations_camera_1,
                            annotations_camera_2,
                            annotations_camera_3,
                        ]
                    ],
                    "patch_metadata": [
                        [
                            {
                                "gs_coords": build_coords(*anno["segmentation"]),
                                "cc_ground_truth": get_cc(),
                                "cc_scene": get_cc(
                                    ground_truth=False, scene=scene, camera=camera + 1
                                ),
                                "shape": get_shape(
                                    im_info["file_name"].split("_")[4].lower()
                                ),  # file_name has format "scene_#_camera_#_<SHAPE>_<HEIGHT>_<DIST>.JPG"
                            }
                            for anno in annos
                            if len(anno["segmentation"]) > 0
                        ][0]
                        for camera, (annos, im_info) in enumerate(
                            zip(
                                [
                                    annotations_camera_1,
                                    annotations_camera_2,
                                    annotations_camera_3,
                                ],
                                [image_camera_1, image_camera_2, image_camera_3],
                            )
                        )
                    ],
                }

                yield_id = yield_id + 1

                patch_size = image_camera_1["file_name"].split(".")[
                    0
                ]  # scene_#_camera_#_<SHAPE>_<HEIGHT>_<DIST>
                patch_size = patch_size.split("_")[-1].lower()  # <DIST>
                if size_dist[size] == patch_size:
                    yield yield_id, example


class DapricotAnnotation(object):
    """Dapricot annotation helper class."""

    def __init__(self, annotation_path):
        with tf.io.gfile.GFile(annotation_path) as f:
            data = json.load(f)
        self._data = data

        # for each images["id"], find all annotations such that annotations["image_id"] == images["id"]
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

    def get_annotations(self, img_id):
        """Return all annotations associated with the image id string."""
        return self._img_id2annotations.get(img_id, [])
