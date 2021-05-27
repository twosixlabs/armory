# Instructions to Integrate TFDS Datasets

1. Get the name, version number of the Tensorflow Dataset, and optionally the config: "name[/config]:version_number", where the brackets denote optional text.
2. Set the environmental variables ARMORY_PRIVATE_S3_ID and ARMORY_PRIVATE_S3_KEY to the appropriate keys with write access to the Armory S3 bucket. 
3. From a locally cloned version of armory, on a new branch, run:
```
python -m armory exec pytorch -- python -m armory.data.integrate_tfds name[/config]:version
```
where the brackets denote optional text.

4. The script will download and process the TFDS dataset, generate TF Records files, create a tarball, and upload the tarball to S3. It also will create a S3 checksum file in ```armory/data/cached_s3_checksums/{name}.txt```
5. Run ```git status``` to confirm the S3 checksum file was generated and to see the path of the template file.
6. Manually put the template code from ```TEMPLATE_{name}.txt``` in ```armory/data/datasets.py```. Create a context object that contains metadata and a preprocessing function that does appropriate integrity checks/input normalizing.
7. [Optional] Add the dataset to SUPPORTED_DATASETS.
8. [Optional] Create a continuous integration test for the dataset in ```tests/test_docker/test_dataset.py```, possibly using ```pytest.skip```.
9. Commit the changes to the branch on your fork of the Armory repo.
10. Open a PR to integrate the dataset.
