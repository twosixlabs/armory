# Adversarial Datasets

The `armory.data.adversarial_datasets` module implements functionality to return adversarial datasets of 
various data modalities. By default, this is a NumPy `ArmoryDataGenerator` which 
implements the methods needed  by the ART framework. Specifically `get_batch` will 
return a tuple of `((data_clean, data_adversarial), labels)` for a specified batch size in numpy format,
where 'data_clean' is a clean example and 'data_adversarial' is the corresponding adversarially attacked example.
Each adversarial dataset contains adversarial examples generated using one or more attacks. To select examples
from a particular attack, specify the desired value for 'adversarial_key' keyword in the scenario configuration -
see table below for valid values for each dataset.

Currently, datasets are loaded using TensorFlow Datasets from cached tfrecord files. 
These tfrecord files will be pulled from S3 if not available on your 
`dataset_dir` directory.

### Usage
To use an adversarial dataset for evaluation, the following keywords in the 'attack' module
of a scenario configuration must be specified. Valid values for each keyword is given in the table below.

Example configuration:

"kwargs": {
    "adversarial_key": "adversarial_univperturbation",
    "batch_size": 1,
    "description": "'adversarial_key' can be 'adversarial_univperturbation' or 'adversarial_univpatch'"
},
"module": "armory.data.adversarial_datasets",
"name": "resisc45_adversarial_224x224",
"type": "preloaded"

### Datasets

| Modality | Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype | adversarial_key |
|:--------: |:----------: |:-----------: |:-------: |:--------: |:--------: |:-------: |
| Image | resisc45_adversarial_224x224 | 
REmote Sensing Image Scene Classification. Contains five images, taken from the test set, of each of the 45 classes, attacked
with targeted universal patch | 
| (N, 224, 224, 3) | uint8 | (N,) | int64 | 'adversarial_univpatch' | 

<br>

### Audio Datasets
| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype | sampling_rate |
|:----------: |:-----------: |:-------: |:--------: |:--------: |:-------: |:-------: |
| [digit](https://github.com/Jakobovski/free-spoken-digit-dataset) | Audio dataset of spoken digits | (N, variable_length) | int64 | (N,) | int64 | 8 kHz |
| [librispeech_dev_clean](http://www.openslr.org/12/) | Librispeech dev dataset for speaker identification  | (N, variable_length)  | int64 | (N,)  | int64 | 16 kHz |

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
