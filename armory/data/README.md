# Datasets

The datasets module implements functionality to download external and internal 
datasets to the armory repository and standardizes them across all evaluations.

### Image Datasets

| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype |
|:---------- |:----------- |:------- |:-------- |:-------- |:------- |
| `cifar10` | CIFAR 10 classes image dataset | (N, 32, 32, 3) | uint8 | (N,) | int64 |
| `german_traffic_sign` | German traffic sign dataset | (N, variable_height, variable_width, 3) | uint8 | (N,) | int64 |
| `imagenet_adversarial` | ILSVRC12 adversarial image dataset from ResNet50 | (N, 224, 224, 3) | float32 | (N,) | int32 |
| `imagenette` | Smaller subset of 10 classes from Imagenet | (N, variable_height, variable_width, 3) | uint8  | (N,) | int64 |
| `mnist` | MNIST hand written digit image dataset | (N, 28, 28, 1) | uint8 | (N,) | int64 |
| `resisc45` | REmote Sensing Image Scene Classification | (N, 256, 256, 3) | uint8 | (N,) | int64 |

### Audio Datasets
| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype |
|:---------- |:----------- |:------- |:-------- |:-------- |:------- |
| `digit` | Audio dataset of spoken digits | (N, variable_length) | int64 | (N,) | int64 |
| `librispeech_dev_clean` | Librispeech dev dataset for speaker identification  | (N, variable_length)  | int64 | (N,)  | int64 |

### Video Datasets
| Dataset    | Description | x_shape | x_dtype  | y_shape  | y_dtype |
|:---------- |:----------- |:------- |:-------- |:-------- |:------- |
| `ucf101` | UCF 101 Action Recognition | (N, variable_frames, 240, 320, 3) | uint8 | (N,) | int64 |


### TF Data Loading
We load datasets using `tf.data` and convert the data to numpy arrays for ingestion in 
any framework. While not the most efficient strategy, this current implementation 
helps us standardize evaluations across all frameworks.

### ArmoryDataSet Generator
*  All DataSets return an `ArmoryDataGenerator` which implements the methods needed 
by the ART framework. Specifically `get_batch` will return a tuple of `(data, labels)` 
for a specficied batch size.
