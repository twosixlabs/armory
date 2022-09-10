"""
Mini speech commands
"""

import os

import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
This is a small excerpt of the [Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)
for use in a tutorial on tensorflow.org. Please refer to the original[dataset](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
for documentation and license information.
"""

_HOMEPAGE = "https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html"

_NUM_CLASSES = 8


_LABELS = [
    "down",
    "up",
    "left",
    "right",
    "go",
    "stop",
    "yes",
    "no",
]


_URL = "https://armory-public-data.s3.us-east-2.amazonaws.com/mini-speech-commands/mini_speech_commands_1.0.0.tar.gz"


class MiniSpeechCommands(tfds.core.GeneratorBasedBuilder):
    """Mini Speech Commands image dataset"""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "audio": tfds.features.Audio(sample_rate=16000),
                    "label": tfds.features.ClassLabel(names=_LABELS),
                    "filename": tfds.features.Text(),
                }
            ),
            supervised_keys=("audio", "label"),
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = os.path.join(
            dl_manager.download_and_extract(_URL), "mini_speech_commands"
        )
        return [
            tfds.core.SplitGenerator(
                name="test",
                gen_kwargs={"path": os.path.join(path, "test")},
            )
        ]

    def _generate_examples(self, path):
        """Yields examples."""

        for label in _LABELS:
            subdir = os.path.join(path, f"{label}")
            for filename in os.listdir(subdir):
                filepath = os.path.join(subdir, filename)
                example = {"audio": filepath, "label": label, "filename": filename}
                yield filepath, example
