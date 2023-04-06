"""carla_video_tracking_test dataset."""

import glob
import os

from PIL import Image
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Synthetic single modality dataset generated using CARLA (https://carla.org).
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

_URLS = "https://armory-public-data.s3.us-east-2.amazonaws.com/carla/carla_video_tracking_test_2.0.0.tar.gz"


class CarlaVideoTrackingTest(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for carla_video_tracking_test dataset."""

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "2.0.0": "Eval 5 CARLA single object tracking data with higher resolution, HD texture, higher frame rate, multiple non-tracked objects, and camera motion",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = tfds.features.FeaturesDict(
            {
                "video": tfds.features.Video(
                    (None, 960, 1280, 3),
                    encoding_format="png",
                ),
                "bboxes": tfds.features.Sequence(
                    tfds.features.Tensor(
                        shape=[4], dtype=tf.int64
                    ),  # ground truth unormalized object bounding boxes given as [x1,y1,x2,y2]
                ),
                # these data only apply to the "green screen patch" objects
                "patch_metadata": tfds.features.FeaturesDict(
                    {
                        "gs_coords": tfds.features.Tensor(
                            shape=[None, 4, 2], dtype=tf.int64
                        ),
                        "masks": tfds.features.Tensor(
                            shape=[None, 960, 1280, 3], dtype=tf.uint8
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
                name="test",
                gen_kwargs={"path": os.path.join(path, "test")},
            )
        ]

    def _generate_examples(self, path):
        """Yields examples."""

        videos = os.listdir(path)
        videos.sort()
        print("videos: {}".format(videos))

        for vi, video in enumerate(videos):
            # Get all frames in a video
            all_frames = glob.glob(
                os.path.join(path, video, "*.png")
            )  # all images including RGB and foreground mask
            mask_frames = glob.glob(
                os.path.join(path, video, "*_mask.png")
            )  # all foreground masks
            rgb_frames = list(set(all_frames) - set(mask_frames))  # all rgb frames

            # sort alphabetically
            rgb_frames.sort()
            mask_frames.sort()

            # verify pairing of RGB and mask
            for r, m in zip(rgb_frames, mask_frames):
                assert r.split(".")[-2] in m

            # get binarized patch masks
            masks = []
            for mf in mask_frames:
                mask = Image.open(os.path.join(path, video, mf)).convert("RGB")
                mask = np.array(mask, dtype=np.uint8)
                mask[np.all(mask == [255, 255, 255], axis=-1)] = 1
                masks.append(mask)

            example = {
                "video": rgb_frames,
                "bboxes": np.load(os.path.join(path, video, "gt_boxes.npy")),
                "patch_metadata": {
                    "gs_coords": np.load(os.path.join(path, video, "gs_coords.npy")),
                    "masks": masks,
                },
            }

            yield vi, example
