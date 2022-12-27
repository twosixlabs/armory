"""german_traffic_sign dataset."""

import tensorflow_datasets as tfds

# TODO(german_traffic_sign): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(german_traffic_sign): BibTeX citation
_CITATION = """
"""


class GermanTrafficSign(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for german_traffic_sign dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(german_traffic_sign): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "label": tfds.features.ClassLabel(names=["no", "yes"]),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(german_traffic_sign): Downloads the data and defines the splits
        path = dl_manager.download_and_extract("https://todo-data-url")

        # TODO(german_traffic_sign): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path / "train_imgs"),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(german_traffic_sign): Yields (key, example) tuples from the dataset
        for f in path.glob("*.jpeg"):
            yield "key", {
                "image": f,
                "label": "yes",
            }
