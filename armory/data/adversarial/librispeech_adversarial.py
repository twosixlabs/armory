"""
TensorFlow Dataset for adversarial librispeech
"""

import os

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
LibriSpeech-dev-clean adversarial audio dataset for SincNet

Universal Perturbation
    Max iterations = 100
    Epsilon = 0.3
    Attacker = Projected Gradient Descent
        Max iterations = 100
        Epsilon = 0.3
        Attack step size = 0.1
        Targeted = false

Projected Gradient Descent
        Max iterations = 100
        Epsilon = 0.3
        Attack step size = 0.1
        Targeted = True
"""

_LABELS = [
    "84",
    "174",
    "251",
    "422",
    "652",
    "777",
    "1272",
    "1462",
    "1673",
    "1919",
    "1988",
    "1993",
    "2035",
    "2078",
    "2086",
    "2277",
    "2412",
    "2428",
    "2803",
    "2902",
    "3000",
    "3081",
    "3170",
    "3536",
    "3576",
    "3752",
    "3853",
    "5338",
    "5536",
    "5694",
    "5895",
    "6241",
    "6295",
    "6313",
    "6319",
    "6345",
    "7850",
    "7976",
    "8297",
    "8842",
]

_URL = (
    "https://armory-public-data.s3.us-east-2.amazonaws.com/adversarial-datasets/"
    "LibriSpeech_SincNet_UnivPerturbation_and_PGD.tar.gz"
)


class LibrispeechAdversarial(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.1.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "audio": {
                        "clean": tfds.features.Audio(),
                        "adversarial_univperturbation": tfds.features.Audio(),
                        "adversarial_perturbation": tfds.features.Audio(),
                    },
                    "label": tfds.features.ClassLabel(names=_LABELS),
                }
            ),
            supervised_keys=("audio", "label"),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators"""
        path = os.path.join(
            dl_manager.download_and_extract(_URL),
            "data",
        )
        splits = [
            tfds.core.SplitGenerator(
                name="adversarial", gen_kwargs={"data_dir_path": path}
            )
        ]
        return splits

    def _generate_examples(self, data_dir_path):
        """Yields examples."""
        split_dirs = [
            "clean",
            "adversarial_univperturbation",
            "adversarial_perturbation",
        ]
        labels = tf.io.gfile.listdir(os.path.join(data_dir_path, split_dirs[0]))
        labels.sort()
        for label in labels:
            chapters = tf.io.gfile.listdir(
                os.path.join(data_dir_path, split_dirs[0], label)
            )
            chapters.sort()

            for chapter in chapters:
                unfiltered_files = tf.io.gfile.listdir(
                    os.path.join(data_dir_path, split_dirs[0], label, chapter)
                )
                clips = [
                    filename for filename in unfiltered_files if ".wav" in filename
                ]
                clips.sort()

                for clip in clips:
                    example = {
                        "audio": {
                            "clean": os.path.join(
                                data_dir_path, split_dirs[0], label, chapter, clip
                            ),
                            "adversarial_univperturbation": os.path.join(
                                data_dir_path, split_dirs[1], label, chapter, clip
                            ),
                            "adversarial_perturbation": os.path.join(
                                data_dir_path, split_dirs[2], label, chapter, clip
                            ),
                        },
                        "label": label,
                    }
                    yield clip, example
