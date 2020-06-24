# Adversarial Datasets

The `armory.data.adversarial_datasets` module implements functionality to return adversarial datasets of 
various data modalities. By default, this is a NumPy `ArmoryDataGenerator` which 
implements the methods needed  by the ART framework. Specifically `get_batch` will 
return a tuple of `((data_clean, data_adversarial), label_clean)` for a specified batch size in numpy format,
where 'data_clean' and 'label_clean' represent a clean example and its true label, and 'data_adversarial'
represents the corresponding adversarially attacked example.
Each adversarial dataset contains adversarial examples generated using one or more attacks.

Currently, datasets are loaded using TensorFlow Datasets from cached tfrecord files. 
These tfrecord files will be pulled from S3 if not available on your 
`dataset_dir` directory.

Refer to [datasets.md](datasets.md) for descriptions of the original datasets from which
the adversarial datasets are created.

### Usage
To use an adversarial dataset for evaluation, specify the desired values for the `name` and `adversarial_key` keywords
in the `attack` module of a scenario configuration. Valid values for each keyword are given in the table below.

Example attack module for image classification scenario:

```json
"attack": {
    "knowledge": "white",
    "kwargs": {
        "adversarial_key": "adversarial_univpatch",
        "batch_size": 1,
        "description": "'adversarial_key' can be 'adversarial_univperturbation' or 'adversarial_univpatch'"
    },
    "module": "armory.data.adversarial_datasets",
    "name": "resisc45_adversarial_224x224",
    "type": "preloaded"
}
```

### Image Datasets
|             `name`             |        `adversarial_key`       |                Description                |               Attack               | Source Split |      x_shape     | x_type | y_shape | y_type |      Size      |
|:------------------------------:|:------------------------------:|:-----------------------------------------:|:----------------------------------:|:------------:|:----------------:|:------:|:-------:|:------:|:--------------:|
| "resisc45_adversarial_224x224" |     "adversarial_univpatch"    | REmote Sensing Image Scene Classification |      Targeted, universal patch     |     test     | (N, 224, 224, 3) |  uint8 |   (N,)  |  int64 | 5 images/class |
| "resisc45_adversarial_224x224" | "adversarial_univperturbation" | REmote Sensing Image Scene Classification | Untargeted, universal perturbation |     test     | (N, 224, 224, 3) |  uint8 |   (N,)  |  int64 | 5 images/class |


### Audio Datasets
|           `name`          | `adversarial_key` |                     Description                    |               Attack               | Source Split |  x_shape  | x_type | y_shape | y_type | sampling_rate |      Size      |
|:-------------------------:|:-----------------:|:--------------------------------------------------:|:----------------------------------:|:------------:|:---------:|:------:|:-------:|:------:|:-------------:|:--------------:|
| "librispeech_adversarial" |   "adversarial"   | Librispeech dev dataset for speaker identification | Untargeted, universal perturbation |     test     | (N, 3000) |  int64 |   (N,)  |  int64 |    16 kHz     | ~5 sec/speaker |


### Video Datasets
|            `name`            |      `adversarial_key`     |         Description        |               Attack               | Source Split |              x_shape              | x_type | y_shape | y_type |      Size      |
|:----------------------------:|:--------------------------:|:--------------------------:|:----------------------------------:|:------------:|:---------------------------------:|:------:|:-------:|:------:|:--------------:|
| "ucf101_adversarial_112x112" |     "adversarial_patch"    | UCF 101 Action Recognition | Untargeted, universal perturbation |     test     | (N, variable_frames, 112, 112, 3) |  uint8 |   (N,)  |  int64 | 5 videos/class |
| "ucf101_adversarial_112x112" | "adversarial_perturbation" | UCF 101 Action Recognition |           Targeted, patch          |     test     | (N, variable_frames, 112, 112, 3) |  uint8 |   (N,)  |  int64 | 5 videos/class |

### Poison Datasets
To be added
