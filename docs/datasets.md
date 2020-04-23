# Datasets

The `armory.data.datasets` module implements functionality to return datasets of 
various data modalities. By default, this is a NumPy `ArmoryDataGenerator` which 
implements the methods needed  by the ART framework. Specifically `get_batch` will 
return a tuple of `(data, labels)` for a specified batch size in numpy format.

We have experimental support for returning `tf.data.Dataset` and 
`torch.utils.data.Dataset`. These can be specified with the `framework` argument to 
the dataset function. Options are `<numpy|tf|pytorch`.

Currently, datasets are loaded using TensorFlow Datasets from cached tfrecord files. 
These tfrecord files will be pulled from S3 if not available on your 
`dataset_dir` directory.

### Image Datasets

| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype |
|:----------: |:-----------: |:-------: |:--------: |:--------: |:-------: |
| [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) | CIFAR 10 classes image dataset | (N, 32, 32, 3) | uint8 | (N,) | int64 |
| [german_traffic_sign](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) | German traffic sign dataset | (N, variable_height, variable_width, 3) | uint8 | (N,) | int64 |
| [imagenette](https://github.com/fastai/imagenette) | Smaller subset of 10 classes from Imagenet | (N, variable_height, variable_width, 3) | uint8  | (N,) | int64 |
| [mnist](http://yann.lecun.com/exdb/mnist/) | MNIST hand written digit image dataset | (N, 28, 28, 1) | uint8 | (N,) | int64 |
| [resisc45](https://arxiv.org/abs/1703.00121) | REmote Sensing Image Scene Classification | (N, 256, 256, 3) | uint8 | (N,) | int64 |
| imagenet_adversarial | ILSVRC12 adversarial dataset from ResNet50 | (N, 224, 224, 3) | uint8 | (N,) | int64 |

<br>

### Audio Datasets
| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype |
|:----------: |:-----------: |:-------: |:--------: |:--------: |:-------: |
| [digit](https://github.com/Jakobovski/free-spoken-digit-dataset) | Audio dataset of spoken digits | (N, variable_length) | int64 | (N,) | int64 |
| [librispeech_dev_clean](http://www.openslr.org/12/) | Librispeech dev dataset for speaker identification  | (N, variable_length)  | int64 | (N,)  | int64 |

<br>

### Video Datasets
| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype |
|:----------: |:-----------: |:-------: |:--------: |:--------: |:-------: |
| [ucf101](https://www.crcv.ucf.edu/data/UCF101.php) | UCF 101 Action Recognition | (N, variable_frames, 240, 320, 3) | uint8 | (N,) | int64 |

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
</style>
