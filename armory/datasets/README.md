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

# Create a new dataset for integration into armory

See the [TFDS v4 CLI](https://www.tensorflow.org/datasets/cli).

On the command line, to create a new standard dataset with name `"my_dataset"`, do:
```
cd armory/datasets/standard
tfds new my_dataset
```
This will populate the `armory/datasets/standard/my_dataset` directory.

At a minimum, you will need to fill in the `my_dataset.py` and `checksums.tsv` files.

## Building and loading

### TFDS standard dataset

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

### Armory standard dataset

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

## Packaging and Uploading for Cache

After a dataset has been successfully built and loaded (locally), it can be packaged and uploaded to the cache.

First, it is recommended that you test the packaging and untarring process without upload/download.

In python:
```
from armory.datasets import package
package.package("my_dataset")  # creates a tar.gz file
package.update("my_dataset")  # adds the tar hash info to "cached_datasets.json"
package.verify("my_dataset")  # uses the "cached_datasets.json" information to verify hash information on tar file
package.extract("my_dataset", overwrite=False)  # This should raise an error, unless you first remove the built dataset; it will ask you to overwrite
package.extract("my_dataset", overwrite=True)  # extracts the tar file into the data directory, overwriting the old one (if overwrite is false, this should raise an error)
```

If you can successfully load the dataset after extracting it here, this part is good.

Now, to upload to s3 (you will need `ARMORY_PRIVATE_S3_ID` and `ARMORY_PRIVATE_S3_KEY`):
```
from armory.datasets import upload
upload.upload("my_dataset")  # this will fail, as you need to explicitly force it to be public
upload.upload("my_dataset", public=True)
```

To download, which will download it directly to the tar cache directory, do:
```
from armory.datasets import download
download.download("my_dataset", overwrite=True, verify=True)
```

You can also download and extract with:
```
from armory.datasets import load
load.ensure_download_extract("my_dataset", verify=True)
```
or just try to load it directly
```
load.load("my_dataset")
```
