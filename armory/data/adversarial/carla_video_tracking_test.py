"""carla_video_tracking_test dataset."""

import os
import glob
import numpy as np
from PIL import Image
import pandas
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Synthetic multimodality (RGB, depth) dataset generated using CARLA (https://carla.org).
"""

_CITATION = """
@inproceedings{Dosovitskiy17,
  title = { {CARLA}: {An} Open Urban Driving Simulator},
  author = {Alexey Dosovitskiy and German Ros and Felipe Codevilla and Antonio Lopez and Vladlen Koltun},
  booktitle = {Proceedings of the 1st Annual Conference on Robot Learning},
  pages = {1--16},
  year = {2017}
}
"""

_URLS = "https://armory-public-data.s3.us-east-2.amazonaws.com/carla/carla_video_tracking_test.tar.gz"


class CarlaVideoTrackingTest(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for carla_video_tracking_test dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = tfds.features.FeaturesDict(
            {
                "video": tfds.features.Video(
                    (None, 600, 800, 3), encoding_format="png",
                ),
                "bboxes": tfds.features.Sequence(
                    tfds.features.Tensor(
                        shape=[4], dtype=tf.int64
                    ),  # ground truth unormalized object bounding boxes given as [x1,y1,x2,y1]
                ),
                # these data only apply to the "green screen patch" objects
                "patch_metadata": tfds.features.FeaturesDict(
                    {
                        "gs_coords": tfds.features.Sequence(
                            tfds.features.Tensor(
                                shape=[2], dtype=tf.int64
                            ),  # green screen vertices in (x,y)
                            length=4,  # always rectangle shape
                        ),
                        "cc_ground_truth": tfds.features.Tensor(
                            shape=[24, 3], dtype=tf.float32
                        ),  # colorchecker color ground truth
                        "cc_scene": tfds.features.Tensor(
                            shape=[24, 3], dtype=tf.float32
                        ),  # colorchecker colors in a scene
                        # binarized segmentation masks of patch.
                        # masks[n,x,y] == 1 indicates patch pixel; 0 otherwise. n is frame number
                        "masks": tfds.features.Sequence(
                            tfds.features.Tensor(shape=[600, 800, 3], dtype=tf.uint8),
                        ),
                    }
                ),
            }
        )

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=features,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(_URLS)

        return [
            tfds.core.SplitGenerator(
                name="test", gen_kwargs={"path": os.path.join(path, "test")},
            )
        ]

    def _generate_examples(self, path):
        """Yields examples."""

        yield_id = 0

        patch_rgb = (
            180,
            130,
            70,
        )  # rgb values of patch object in semantic segmentation image

        videos = os.listdir(path)
        videos.sort()

        for video in videos:
            # Get all frames in a video
            all_frames = glob.glob(
                os.path.join(path, video, "*.png")
            )  # all images including RGB and semantic segmentation
            ss_frames = glob.glob(
                os.path.join(path, video, "*SS.png")
            )  # all semantic segmentaiton frames
            rgb_frames = list(set(all_frames) - set(ss_frames))  # all rgb frames

            # sort alphabetically
            rgb_frames.sort()
            ss_frames.sort()

            # verify pairing of RGB and SS
            for r, s in zip(rgb_frames, ss_frames):
                assert r.split(".")[-2] in s

            # get images
            rgb_imgs = []
            for im in rgb_frames:
                img_rgb = Image.open(os.path.join(path, video, im)).convert("RGB")
                img_rgb = np.array(img_rgb)
                rgb_imgs.append(img_rgb)

            # get binarized patch masks
            masks = []
            for ss in ss_frames:
                img_ss = Image.open(os.path.join(path, video, ss)).convert("RGB")
                img_ss = np.array(img_ss)
                mask = np.zeros_like(img_ss)
                mask[np.all(img_ss == patch_rgb, axis=-1)] = 1
                masks.append(mask)

            # Get ground truth bounding box for each frame and convert to new format
            bbfile = os.path.join(path, video, "gt_mask_carla.txt")
            bboxes_orig = np.loadtxt(bbfile, delimiter=",")  # format [x,y,w,h]
            bboxes = []  # expected format [x1,y1,x2,y2]
            for bbox in bboxes_orig:
                bbox_t = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                bboxes.append(bbox_t)

            # convert patch mask to patch coordinates [top_left, top_right, bottom_right, bottom_left]
            def build_coords(mask):
                """
        Get the corner points of a patch by using its segmentation mask.

        Arguments:
            mask: A numpy array of shape (height, width). mask will be converted
                  to a uint8 image for use with the cornerHarris algorithm

        Returns:
            pts: The corner points (vertices) of the mask/patch. This is a list
                of lists, with each nested list containing the (x,y) coordinate
                of a vertex/corner. The points will be in the order of:
                    [Top/Left, Top/Right, Bottom/Right, Bottom/Left]
        """
                # Importing cv2 inside function, since not all twosixarmory images contain cv2 package
                import cv2

                if mask.dtype is not np.uint8 or max.max() <= 1.0:
                    mask = (255 * mask).astype(np.uint8)
                mask_area = np.sum(mask > 0)
                min_dist = np.sqrt(mask_area) / 5
                # Threshold can change based on lighting and position, search adaptively
                for thresh in np.arange(1, 0.05, -0.05):
                    corners = cv2.goodFeaturesToTrack(mask, 4, thresh, min_dist)
                    if corners is None:
                        continue
                    elif len(corners) < 4:
                        continue
                    else:
                        pts = corners.squeeze()
                        break
                pts_by_dist = np.arctan2(
                    pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean()
                )
                sorted_idxs = np.argsort(pts_by_dist)
                pts = pts[sorted_idxs]
                return pts.astype(np.int)

            # get colorchecker color box values. There are 24 color boxes, so output shape is (24, 3)
            def get_cc(ground_truth=True):
                if ground_truth:
                    return (
                        pandas.read_csv(
                            os.path.join(
                                path, video, "xrite_passport_colors_sRGB-GMB-2005.csv",
                            ),
                            header=None,
                        )
                        .to_numpy()
                        .astype("float32")
                    )
                else:
                    return (
                        pandas.read_csv(
                            os.path.join(path, video, "CC.csv",), header=None,
                        )
                        .to_numpy()
                        .astype("float32")
                    )

            patch_coords = build_coords(masks[0][:, :, 0:1])

            example = {
                "video": rgb_imgs,
                "bboxes": bboxes,
                "patch_metadata": {
                    "gs_coords": patch_coords,
                    "cc_ground_truth": get_cc(),
                    "cc_scene": get_cc(ground_truth=False),
                    "masks": masks,
                },
            }

            yield_id = yield_id + 1
            yield yield_id, example
