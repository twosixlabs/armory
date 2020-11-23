# Datasets

The `armory.data.datasets` module implements functionality to return datasets of 
various data modalities. By default, this is a NumPy `ArmoryDataGenerator` which 
implements the methods needed  by the ART framework. Specifically `get_batch` will 
return a tuple of `(data, labels)` for a specified batch size in numpy format.

We have experimental support for returning `tf.data.Dataset` and 
`torch.utils.data.Dataset`. These can be specified with the `framework` argument to 
the dataset function. Options are `<numpy|tf|pytorch>`.

Currently, datasets are loaded using TensorFlow Datasets from cached tfrecord files. 
These tfrecord files will be pulled from S3 if not available on your 
`dataset_dir` directory.

### Image Datasets

| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype | splits |
|:----------: |:-----------: |:-------: |:--------: |:--------: |:-------: |:------: |
| [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) | CIFAR 10 classes image dataset | (N, 32, 32, 3) | float32 | (N,) | int64 | train, test |
| [german_traffic_sign](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) | German traffic sign dataset | (N, variable_height, variable_width, 3) | float32 | (N,) | int64 | train, test | 
| [imagenette](https://github.com/fastai/imagenette) | Smaller subset of 10 classes from Imagenet | (N, variable_height, variable_width, 3) | uint8  | (N,) | int64 | train, validation |
| [mnist](http://yann.lecun.com/exdb/mnist/) | MNIST hand written digit image dataset | (N, 28, 28, 1) | float32 | (N,) | int64 | train, test | 
| [resisc45](https://arxiv.org/abs/1703.00121) | REmote Sensing Image Scene Classification | (N, 256, 256, 3) | float32 | (N,) | int64 | train, validation, test | 
| [xView](https://arxiv.org/pdf/1802.07856) | Objects in Context in Overhead Imagery | (N, variable_height, variable_width, 3) | float32 | n/a | dict | train, test | 

<br>

### Audio Datasets
| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype | sampling_rate | splits |
|:----------: |:-----------: |:-------: |:--------: |:--------: |:-------: |:-------: |:------: |
| [digit](https://github.com/Jakobovski/free-spoken-digit-dataset) | Audio dataset of spoken digits | (N, variable_length) | int64 | (N,) | int64 | 8 kHz | train, test |
| [librispeech](http://www.openslr.org/12/) | Librispeech dataset for automatic speech recognition  | (N, variable_length)  | float32 | (N,)  | bytes | 16 kHz | dev_clean, dev_other, test_clean, train_clean100 |
| [librispeech-full](http://www.openslr.org/12/) | Full Librispeech dataset for automatic speech recognition | (N, variable_length)  | float32 | (N,)  | bytes | 16 kHz | dev_clean, dev_other, test_clean, train_clean100, train_clean360, train_other500 |
| [librispeech_dev_clean](http://www.openslr.org/12/) | Librispeech dev dataset for speaker identification  | (N, variable_length)  | float32 | (N,)  | int64 | 16 kHz | train, validation, test |
| [librispeech_dev_clean_asr](http://www.openslr.org/12) | Librispeech dev dataset for automatic speech recognition | (N, variable_length) | float32 | (N,) | bytes | 16 kHz | train, validation, test |

NOTE: because the Librispeech dataset is over 300 GB with all splits, the ```librispeech_full``` dataset has
all splits, whereas the ```librispeech``` dataset does not have the train_clean360 or train_other500 splits.
<br>

### Video Datasets
| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype | splits |
|:----------: |:-----------: |:-------: |:--------: |:--------: |:-------: |:------: |
| [ucf101](https://www.crcv.ucf.edu/data/UCF101.php) | UCF 101 Action Recognition | (N, variable_frames, None, None, 3) | float32 | (N,) | int64 | train, test |
| [ucf101_clean](https://www.crcv.ucf.edu/data/UCF101.php) | UCF 101 Action Recognition | (N, variable_frames, None, None, 3) | float32 | (N,) | int64 | train, test |

NOTE: The dimension of UCF101 videos is `(N, variable_frames, 240, 320, 3)` for the entire training set and all of the test set except for 4 examples.
For those, the dimensions are `(N, variable_frames, 226, 400, 3)`. If not shuffled, these correspond to (0-indexed) examples 333, 694, 1343, and 3218.
NOTE: The only difference between `ucf101` and `ucf101_clean` is that the latter uses the ffmpeg flag `-q:v 2`, which results in fewer video compression errors.These are stored as separate datasets, however.

<br>

### Multimodal Datasets
| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype | splits |
|:----------: |:-----------: |:-------: |:--------: |:--------: |:-------: |:------: |
| [so2sat](https://mediatum.ub.tum.de/1454690) | Co-registered synthetic aperture radar and multispectral optical images | (N, 32, 32, 14) | float32 | (N,) | int64 | train, validation |

<br>

### Preprocessing

Armory applies preprocessing to convert each dataset to canonical form (e.g. normalize the range of values, set the data type).
The poisoning scenario loads its own custom preprocessing, however the GTSRB data is also available in its canonical form.
Any additional preprocessing that is desired should occur as part of the model under evaluation.

Canonical preprocessing is not yet supported when `framework` is `tf` or `pytorch`.

### Splits

Datasets that are imported directly from TFDS have splits that are defined according to the
Tensorflow Datasets [library](https://www.tensorflow.org/datasets/catalog/overview). The
`german-traffic-sign` dataset split follows the description of the original source of the
[dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The `digits`
 dataset split follows the description of the original source of the 
 [dataset](https://github.com/Jakobovski/free-spoken-digit-dataset#usage). The following
 table describes datasets with custom splits in Armory.

|        Dataset        |    Split   |               Description              |                   Split logic details                  |
|:---------------------:|:----------:|:--------------------------------------:|:------------------------------------------------------:|
|       resisc_45       |    train   |         First 5/7 of dataset           | See armory/data/resisc45/resisc45_dataset_partition.py |
|                       | validation |          Next 1/7 of dataset           |                                                        |
|                       |    test    |         Final 1/7 of dataset           |                                                        |
| librispeech_dev_clean |    train   | 1371 recordings from dev_clean dataset |   Assign discrete clips so at least 50% of audio time  |
|                       | validation |  692 recordings from dev_clean dataset |       is in train, at least 25% is in validation,      |
|                       |    test    |  640 recordings from dev_clean dataset |              and the remainder are in test             |
| xView                 | train      | ~58k images                            |     see [xView arXiv](https://arxiv.org/abs/1802.07856)     |
|                       | test       | ~18k images                            |                                                        |


<br>


### Adversarial Datasets
See [adversarial_datasets.md](adversarial_datasets.md) for descriptions of adversarial examples created from some of the datasets listed here.

### Dataset Licensing
See [dataset_licensing.md](dataset_licensing.md) for details related to the licensing of datasets.


<br>
<style>
    table th:first-of-type {
    width: 10%;
}
table th:nth-of-type(2) {
    width: 50%;
}
table th:nth-of-type(3) {
    width: 30%;
}
table th:nth-of-type(4) {
    width: 10%;
}
table th:nth-of-type(5) {
    width: 10%;
}
</style>
