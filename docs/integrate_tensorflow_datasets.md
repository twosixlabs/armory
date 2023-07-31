# Instructions to Integrate TFDS Datasets

1. Get the name, version number of the Tensorflow Dataset, and optionally the config: "name[/config]:version_number", where the brackets denote optional text.
2. Set the environmental variables ARMORY_PRIVATE_S3_ID and ARMORY_PRIVATE_S3_KEY to the appropriate keys with write access to the Armory S3 bucket.
3. From a locally cloned version of armory, on a new branch, run:
```
python -m armory exec pytorch -- python -m armory.data.integrate_tfds name[/config]:version
```
where the brackets denote optional text.

The script will download and process the TFDS dataset, generate TF Records files, create a tarball, and upload the tarball to S3. It also will create a S3 checksum file in ```armory/data/cached_s3_checksums/{name}.txt```

4. Run ```git status``` to confirm the S3 checksum file was generated and to see the path of the template file.
5. Manually put the template code from ```TEMPLATE_{name}.txt``` in ```armory/data/datasets.py```. Create a context object that contains metadata and a preprocessing function that does appropriate integrity checks/input normalizing. See for example the [canonical fixed-size image preprocessing function](https://github.com/twosixlabs/armory/blob/deb7a469bf4a7497d14fdd87eba6417b5e44589f/armory/data/datasets.py#L617-L631) which checks the shapes of an image, and renormalizes it to be in the appropriate range defined by the context object (typically 0.0-1.0) with a standard type. See the documentation on dataset [preprocessing](https://github.com/twosixlabs/armory/docs/datasets.md) for more details.
6. [Optional] Add the dataset to the [SUPPORTED_DATASETS](https://github.com/twosixlabs/armory/blob/deb7a469bf4a7497d14fdd87eba6417b5e44589f/armory/data/datasets.py#L1498-L1511) dictionary by adding a key with the dataset's name and value of the dataset function from the template code.
7. [Optional] Create a continuous integration test for the dataset in ```tests/test_docker/test_dataset.py```, possibly using ```pytest.skip```.
8. Commit the changes to the branch on your fork of the Armory repo.
9. Open a PR to integrate the dataset.
