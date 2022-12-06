"""
German traffic sign dataset with 43 classes and over 50,000 images.
"""

import csv
import os

import PIL
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
German traffic sign dataset with 43 classes and over 50,000 images.
"""

_HOMEPAGE = "http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset"

_NUM_CLASSES = 43
_LABELS = [str(x) for x in range(_NUM_CLASSES)]

_URL = "https://armory-public-data.s3.us-east-2.amazonaws.com/german-traffic-sign/german_traffic_sign.tar.gz"


class GermanTrafficSign(tfds.core.GeneratorBasedBuilder):
    """German traffic sign image dataset"""

    VERSION = tfds.core.Version("3.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=[None, None, 3]),
                    "label": tfds.features.ClassLabel(names=_LABELS),
                    "filename": tfds.features.Text(),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = os.path.join(dl_manager.download_and_extract(_URL), "GTSRB")
        splits = [
            tfds.core.SplitGenerator(name=x, gen_kwargs={"path": path, "split": x})
            for x in (tfds.Split.TRAIN, tfds.Split.TEST)
        ]
        return splits

    def _generate_examples(self, path, split):
        """Yields examples. Converts PPM files to BMP before yielding."""

        def _read_images(prefix, gtFile):
            with open(gtFile, newline="") as csvFile:
                gtReader = csv.reader(csvFile, delimiter=";")
                next(gtReader)  # skip header
                # loop over all images in current annotations file
                for i, row in enumerate(gtReader):
                    ppm_filename = row[0]
                    ppm_filepath = os.path.join(prefix, ppm_filename)
                    # translate ppm files to bmp files
                    base, ext = os.path.splitext(ppm_filename)
                    bmp_filename = base + ".bmp"
                    bmp_filepath = os.path.join(prefix, bmp_filename)
                    with PIL.Image.open(ppm_filepath) as image:
                        image.save(bmp_filepath, "BMP")

                    example = {
                        "image": bmp_filepath,
                        "label": row[7],
                        "filename": bmp_filename,
                    }
                    yield bmp_filepath, example

        if split is tfds.Split.TRAIN:
            for c in range(_NUM_CLASSES):
                # subdirectory for class
                prefix = os.path.join(path, "Final_Training", "Images", f"{c:05d}")
                # annotations file
                gtFile = os.path.join(prefix, f"GT-{c:05d}.csv")
                for x in _read_images(prefix, gtFile):
                    yield x
        elif split is tfds.Split.TEST:
            prefix = os.path.join(path, "Final_Test", "Images")
            gtFile = os.path.join(path, "GT-final_test.csv")
            for x in _read_images(prefix, gtFile):
                yield x
        else:
            raise ValueError(f"split {split} not in ('train', 'test')")
