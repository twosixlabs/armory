"""UCF101 with video compression artifacts removed"""

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.video import ucf101


class Ucf101Clean(ucf101.Ucf101):
    def _info(self):
        if self.builder_config.width is not None:
            if self.builder_config.height is None:
                raise ValueError("Provide either both height and width or none.")
            ffmpeg_extra_args = (
                "-q:v" "2",
                "-vf",
                "scale={}x{}".format(
                    self.builder_config.height, self.builder_config.width
                ),
            )
        else:
            ffmpeg_extra_args = ("-q:v", "2")

        video_shape = (None, self.builder_config.height, self.builder_config.width, 3)
        labels_names_file = tfds.core.get_tfds_path(ucf101._LABELS_FNAME)
        features = tfds.features.FeaturesDict(
            {
                "video": tfds.features.Video(
                    video_shape,
                    ffmpeg_extra_args=ffmpeg_extra_args,
                    encoding_format="jpeg",
                ),
                "label": tfds.features.ClassLabel(names_file=labels_names_file),
            }
        )
        return tfds.core.DatasetInfo(
            builder=self,
            description="A 101-label video classification dataset.",
            features=features,
            homepage="https://www.crcv.ucf.edu/data/UCF101.php",
            citation=ucf101._CITATION,
        )
