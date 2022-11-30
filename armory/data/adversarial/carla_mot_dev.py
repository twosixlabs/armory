"""carla_mot_dev dataset."""

import glob
import os
import re

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

_URLS = "https://armory-public-data.s3.us-east-2.amazonaws.com/carla/carla_mot_dev_1.0.0.tar.gz"


class CarlaMOTDev(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for carla_mot_dev dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = tfds.features.FeaturesDict(
            {
                "video": tfds.features.Video(
                    (None, 960, 1280, 3),
                    encoding_format="png",
                ),
                "video_name": tfds.features.Text(),
                # ground truth annotation is a 2D NDArray where each row represents a detection with format:
                # <timestep> <object_id> <bbox top-left x> <bbox top-left y> <bbox width> <bbox height> <confidence_score=1> <class_id> <visibility=1>
                "annotations": tfds.features.Tensor(shape=[None, 9], dtype=tf.float32),
                # these data only apply to the "green screen patch" objects
                "patch_metadata": tfds.features.FeaturesDict(
                    {
                        "gs_coords": tfds.features.Tensor(
                            shape=[None, 4, 2], dtype=tf.int64
                        ),
                        "masks": tfds.features.Video(
                            (None, 960, 1280, 3),
                            encoding_format="png",
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
                name="dev",
                gen_kwargs={"path": os.path.join(path, "dev")},
            )
        ]

    def _generate_examples(self, path):
        """Yields examples."""

        videos = os.listdir(path)
        videos.sort()

        for vi, video in enumerate(videos):
            # Get all data
            rgb_frames = glob.glob(os.path.join(path, video, "rgb", "*.png"))
            mask_frames = glob.glob(
                os.path.join(path, video, "foreground_mask", "*.png")
            )
            gs_coords_frames = glob.glob(
                os.path.join(path, video, "patch_metadata", "*.npy")
            )
            annotations_txt = os.path.join(path, video, "instances.txt")

            # sort alphabetically
            rgb_frames.sort()
            mask_frames.sort()
            gs_coords_frames.sort()

            # verify data consistency
            assert len(rgb_frames) == len(mask_frames) == len(gs_coords_frames)
            for r, m, g in zip(rgb_frames, mask_frames, gs_coords_frames):
                r = re.split("[./]", r)[-2]  # get filename without extension
                m = re.split("[./]", m)[-2]
                g = re.split("[./]", g)[-2]
                assert r == m == g

            gs_coords = create_gs_coords(gs_coords_frames)

            annotations = np.loadtxt(annotations_txt, dtype=np.float32)

            example = {
                "video": rgb_frames,
                "video_name": video,
                "annotations": annotations,
                "patch_metadata": {
                    "gs_coords": gs_coords,
                    "masks": mask_frames,
                },
            }

            yield vi, example


def create_gs_coords(gs_coords_files):
    """
    Convert a list of .npy files, each storing greenscreen coordinates
    for one frame, to a 3D NDArray
    """
    gs_coords = []
    for f in gs_coords_files:
        gs_coords.append(np.load(f))

    return np.array(gs_coords, dtype=np.int)
