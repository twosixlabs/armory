"""
TensorFlow Dataset for resisc45 with train/validate/test splits
"""

import itertools
import os

import tensorflow_datasets.public_api as tfds

_SAMPLE_RATE = 8000
_MIN_LENGTH = 1148
_MAX_LENGTH = 18262

_DESCRIPTION = f"""\
An audio dataset of spoken digits, with three users.

Original data is in .wav format with int16 values, sampled at {_SAMPLE_RATE} Hz.
Min length: {_MIN_LENGTH} samples
Max length: {_MAX_LENGTH} samples
"""

_DIGITS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

_USERS = [
    "jackson",
    "nicolas",
    "theo",
]

_HOMEPAGE = "https://github.com/Jakobovski/free-spoken-digit-dataset"

_URL = "https://armory-public-data.s3.us-east-2.amazonaws.com/digit/digit-1.0.8.tar.gz"


class Digit(tfds.core.GeneratorBasedBuilder):
    """Audio 'MNIST' dataset, with spoken digits 0 to 9"""

    VERSION = tfds.core.Version("1.0.8")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "audio": tfds.features.Audio(),
                    "label": tfds.features.ClassLabel(names=_DIGITS),
                    "filename": tfds.features.Text(),
                }
            ),
            supervised_keys=("audio", "label"),
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = os.path.join(dl_manager.download_and_extract(_URL), "recordings")
        splits = [
            tfds.core.SplitGenerator(name=x, gen_kwargs={"path": path, "split": x})
            for x in (tfds.Split.TRAIN, tfds.Split.TEST)
        ]
        return splits

    def _generate_examples(self, path, split):
        """Yields examples."""
        if split is tfds.Split.TRAIN:
            samples = range(5, 50)
        elif split is tfds.Split.TEST:
            samples = range(5)
        else:
            raise ValueError(f"split {split} not in ('train', 'test')")

        for digit, user, sample in itertools.product(_DIGITS, _USERS, samples):
            filename = f"{digit}_{user}_{sample}.wav"
            filepath = os.path.join(path, filename)
            example = {
                "audio": filepath,
                "label": digit,
                "filename": filename,
            }
            yield filepath, example
