"""
TensorFlow Datasets requires a lot of work to generate checksums.
    Specifically, you need to have TFDS cloned locally,
    have your file importable locally from TFDS, and then run a script from the repo.
    There does not appear to be a programmatic way of doing it with simple
    imports from TFDS.

This fills in the gap. A checksums file should be a txt file with one line per file.

Each line has <name> <size in bytes> <sha256> in a space-separated manner.
    Unclear what to do if <name> has spaces in it

Key: lines should be in alphabetical order based on <name>
"""

import os

from armory.data import datasets, utils


class Builder:
    """
    Builder for checksum files.
    dataset_name - must be the name of the file that includes tfds.core.GeneratorBasedBuilder

    Example usage for how the resisc45 dataset checksum was generated:
        >>> builder = tfds_checksum.Builder("resisc45_split",
            "https://armory-public-data.s3.us-east-2.amazonaws.com/resisc45",
            "/local/path/to/resisc45/directory",
            ["resisc45_train.tar.gz", "resisc45_test.tar.gz",
                "resisc45_validation.tar.gz"])
        >>> builder.compute()
        >>> builder.dump()
    """

    def __init__(self, dataset_name, base_url, base_filedir, archive_names):
        self.dataset_name = dataset_name
        archive_names = sorted(archive_names)
        self.urls = [os.path.join(base_url, x) for x in archive_names]
        self.filepaths = [os.path.join(base_filedir, x) for x in archive_names]
        self.values = None

    def compute(self):
        values = []
        for url, filepath in zip(self.urls, self.filepaths):
            size, sha256 = compute_file(filepath)
            values.append((url, str(size), sha256))
        self.values = values

    def dumps(self):
        if self.values is None:
            raise ValueError("Must first run 'compute'")
        lines = [" ".join(x) + "\n" for x in self.values]
        return "".join(lines)

    def dump(self, checksum_filepath=None, overwrite=False):
        if checksum_filepath is None:
            checksum_filepath = os.path.join(
                datasets.CHECKSUMS_DIR, self.dataset_name + ".txt"
            )
        if os.path.exists(checksum_filepath) and not overwrite:
            raise ValueError(f"{checksum_filepath} already exists. Set overwrite=True")
        with open(checksum_filepath, "w") as f:
            f.write(self.dumps())


def compute_file(filepath):
    size = os.path.getsize(filepath)
    sha256 = utils.sha256(filepath)
    return size, sha256
