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
| [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) | CIFAR 10 classes image dataset | (N, 32, 32, 3) | uint8 | (N,) | int64 | train, test |
| [german_traffic_sign](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) | German traffic sign dataset | (N, variable_height, variable_width, 3) | uint8 | (N,) | int64 | train, test | 
| [imagenette](https://github.com/fastai/imagenette) | Smaller subset of 10 classes from Imagenet | (N, variable_height, variable_width, 3) | uint8  | (N,) | int64 | train, validation |
| [mnist](http://yann.lecun.com/exdb/mnist/) | MNIST hand written digit image dataset | (N, 28, 28, 1) | uint8 | (N,) | int64 | train, test | 
| [resisc45](https://arxiv.org/abs/1703.00121) | REmote Sensing Image Scene Classification | (N, 256, 256, 3) | uint8 | (N,) | int64 | train, validation, test | 
| imagenet_adversarial | ILSVRC12 adversarial dataset from ResNet50 | (N, 224, 224, 3) | uint8 | (N,) | int64 | NA |
| [xView](https://arxiv.org/pdf/1802.07856) | Objects in Context in Overhead Imagery | (N, variable_height, variable_width, 3) | uint8 | n/a | dict | train, test | 

<br>

### Audio Datasets
| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype | sampling_rate | splits |
|:----------: |:-----------: |:-------: |:--------: |:--------: |:-------: |:-------: |:------: |
| [digit](https://github.com/Jakobovski/free-spoken-digit-dataset) | Audio dataset of spoken digits | (N, variable_length) | int64 | (N,) | int64 | 8 kHz | train, test |
| [librispeech_dev_clean](http://www.openslr.org/12/) | Librispeech dev dataset for speaker identification  | (N, variable_length)  | int64 | (N,)  | int64 | 16 kHz | train, validation, test |

<br>

### Video Datasets
| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype | splits |
|:----------: |:-----------: |:-------: |:--------: |:--------: |:-------: |:------: |
| [ucf101](https://www.crcv.ucf.edu/data/UCF101.php) | UCF 101 Action Recognition | (N, variable_frames, 240, 320, 3) | uint8 | (N,) | int64 | train, test |

<br>

### Preprocessing

Input-modifying preprocessing of datasets occurs as part of a model used within Armory. The cached
datasets are preprocessed into tfrecords, however this preprocessing primarily consists of changing the
representation of inputs, e.g. running pydub on flac audio files.

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
