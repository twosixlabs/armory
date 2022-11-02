# Setup

This is designed to be run in an armory pytorch container.
Recommended use:
```
armory launch pytorch
```
Then work in that environment.

# Structure

The code exists in `armory/datasets`, with two subdirectories:
- `standard` - non-adversarial datasets
- `adversarial` - adversarial datasets

## Create a new dataset for integration into armory

See the [TFDS v4 CLI](https://www.tensorflow.org/datasets/cli).

On the command line, to create a new standard dataset with name `"my_dataset"`, do:
```
cd armory/datasets/standard
tfds new my_dataset
```
This will populate the `armory/datasets/standard/my_dataset` directory.

At a minimum, you will need to fill in the `my_dataset.py` and `checksums.tsv` files.

# Examples

## TFDS standard dataset

MNIST is a standard built-in TFDS dataset. To build this, do:
```
python -m armory.datasets.build mnist
```

To load it, you can do:
```
python -m armory.datasets.load mnist
```

Or, in an interpreter, you can do:
```
from armory.datasets import load
info, ds = load.load("mnist")
```

## Armory standard dataset

Digit is a basic audio dataset that we created the code for. To build this, do:
```
python -m armory.datasets.build digit [--overwrite] [--register_checksums]
```
NOTE: When building a dataset for the first time, use `--register_checksums` to populate the `checksums.tsv` file.

Alternatively, in an interpreter, you can do:
```
from armory.datasets import build
build.build("digit")
```

To load it, you can do:
```
python -m armory.datasets.load digit
```

Or, in an interpreter, you can do:
```
from armory.datasets import load
info, ds = load.load("digit")
# Alternatively:
info, ds = load.from_directory("/armory/datasets/new_builds/digit/1.0.8")
```
