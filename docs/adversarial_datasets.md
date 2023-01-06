# Adversarial Datasets

The `armory.data.adversarial_datasets` module implements functionality to return adversarial datasets of
various data modalities. By default, this is a NumPy `ArmoryDataGenerator` which
implements the methods needed  by the ART framework.

There are two kinds of adversarial datasets in Armory: *preloaded* as well as *green-screen*. Preloaded datasets contain
examples with universal adversarial perturbations. For preloaded adversarial datasets, `get_batch()` returns
a tuple of `((data_clean, data_adversarial), label_clean)` for a specified batch size in numpy format,
where `data_clean` and `label_clean` represent a clean example and its true label, and `data_adversarial`
represents the corresponding adversarially attacked example. The lone exception is the APRICOT dataset, which is preloaded
but returns a tuple of `(data_adversarial, label_adversarial)` as the images don't have benign counterparts.

The green-screen adversarial datasets in Armory are DAPRICOT and CARLA. Each image in these datasets contains a
green-screen, for which adversarial patches generated during an attack are inserted onto. For these datasets
`get_batch()` returns a tuple of `(data_adversarial, (objects_label, green_screen_label))`.



Currently, datasets are loaded using TensorFlow Datasets from cached tfrecord files.
If the files are not already present locally in your `dataset_dir` directory, Armory will download them
from Two Six's public S3 dataset repository.



### Green-screen Image and Video Datasets
|               `name`               |                   `split`                    |                Description                | Source Split |               x_shape               | x_type | y_shape |                Size                 |
|:----------------------------------:|:--------------------------------------------:|:-----------------------------------------:|:------------:|:-----------------------------------:|:------:|:-------:|:-----------------------------------:|
|     "dapricot_dev_adversarial"     | ["small", medium", "large", "adversarial"] * | Physical Adversarial Attacks on Object Detection|     dev      |        (nb, 3, 1008, 756, 3)        | uint8  | 2-tuple | 81 examples (3 images per example)  |
|    "dapricot_test_adversarial"     | ["small", medium", "large", "adversarial"] * | Physical Adversarial Attacks on Object Detection|     test     |        (nb, 3, 1008, 756, 3)        | uint8  | 2-tuple | 324 examples (3 images per example) |
|        "carla_obj_det_dev"         |                   ["dev"]                    | [CARLA Simulator Object Detection](https://carla.org) |     dev      |      (nb=1, 960, 1280, 3 or 6)      | uint8  | 2-tuple |              31 images              |
|        "carla_obj_det_test"        |                   ["test"]                   | [CARLA Simulator Object Detection](https://carla.org) |     test     |    (nb=1, 960, 1280, 3 or 6)      | uint8  | 2-tuple |              30 images              |
|      "carla_over_obj_det_dev"      |                   ["dev"]                    | [CARLA Simulator Object Detection](https://carla.org) |     dev      |      (nb=1, 960, 1280, 3 or 6)      | uint8  | 2-tuple |              20 images              |                            
|      "carla_over_obj_det_test"     |                   ["test"]                   | [CARLA Simulator Object Detection](https://carla.org) |     test      |      (nb=1, 960, 1280, 3 or 6)      |uint8  | 2-tuple |              15 images              |
|     "carla_video_tracking_dev"     |                   ["dev"]                    | [CARLA Simulator Video Tracking](https://carla.org) |     dev      |  (nb=1, num_frames, 960, 1280, 3)   | uint8  | 2-tuple |              20 videos              |
|    "carla_video_tracking_test"     |                   ["test"]                   | [CARLA Simulator Video Tracking](https://carla.org) |     test     | (nb=1, num_frames, 960, 1280, 3)     | uint8  | 2-tuple |              20 videos              |
| "carla_multi_object_tracking_dev"  |                   ["dev"]                    | [CARLA Simulator Multi-object Video Tracking](https://carla.org) |     dev      | (nb=1, num_frames, 960, 1280, 3) | uint8  | 2-tuple |              20 videos              |
| "carla_multi_object_tracking_test" |                   ["test"]                   | [CARLA Simulator Multi-object Video Tracking](https://carla.org) |     test     | (nb=1, num_frames, 960, 1280, 3) | uint8  | 2-tuple |              10 videos              |

\* the "small" split, for example, is the subset of images containing small patch green-screens. Using the "adversarial" split returns the entire dataset.

##### CARLA Multi-Object Tracking
The ground truth annotation for CARLA multi-object tracking is a 2D NDArray where each row represents a detection with format: `<timestep> <object id> <bbox top left x> <bbox top left y> <bbox width> <bbox height> <confidence score=1> <class id> <visibility=1>`.  By nature of this dataset, there may be multiple objects present at each timestep; therefore, each object is assigned an ID so that all annotations corresponding to a given object can be identified.
- `timestep`: The timestep indicates which frame this annotation belongs to; indexing begins at 1.
- `object id`: An id referring to the unique object in the frame that this annotation describes.
- `bbox top left x`: The top left x coordinate of the bounding box of this object.
- `bbox top left y`: The top left y coordinate of the bounding box of this object.
- `bbox width`: The width of the bounding box of this object in pixels.
- `bbox height`: The height of the bounding box of this object in pixels.
- `confidence score`: 1 for ground truth objects; models may output values between 0 and 1 to describe the confidence of their predictions.
- `class id`: The class label of this object.
- `visibility`: 1 for all annotations in this dataset.  An object that is not visible will not have an annotation for that timestep.


##### D-APRICOT
The D-APRICOT dataset does NOT contain labels/bounding boxes for COCO objects, which may occasionally appear in the
background (e.g. car). Each image contains one green screen intended for patch insertion. The green screen shapes vary
between diamond, rectangle, and octagon. A dataset example consists of three images, each of a different camera
 angle of the same scene and green screen. The intended threat model is a targeted attack where the inserted patch
is meant to induce the model to predict a specific class at the location of the patch.


##### CARLA Object Detection
The carla_obj_det_dev and carla_obj_det_test datasets contain rgb and depth modalities. The modality defaults to `"rgb"` and must be one of `["rgb", "depth", "both"]`.
When using the dataset function imported from [armory.data.adversarial_datasets](../armory/data/adversarial_datasets.py), this value is passed via the `modality` kwarg. When running an Armory scenario, the value is specified in the dataset_config as such:
```json
 "dataset": {
    "batch_size": 1,
    "modality": "rgb",
}
```
When `modality` is set to `"both"`, the input will be of shape `(nb=1, num_frames, 960, 1280, 6)` where `x[..., :3]` are
the rgb channels and `x[..., 3:]` the depth channels.  The depth information is encoded on a log scale in grayscale format (all three depth channels are the same).

The carla_over_obj_det_dev dataset has the same properties as the above mentioned datasets but is collected utilizing overhead perspectives.  In addition, the depth channels are encoded with an [RGB encoding](https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera).  This depth format can be converted to linear grayscale or log grayscale (e.g. for human viewing) using the utility functions in [carla_obj_det_utils.py](../armory/art_experimental/attacks/carla_obj_det_utils.py)

### Usage of Preloaded Adversarial Datasets
To use a preloaded adversarial dataset for evaluation, set `attack_config["type"]` to `"preloaded"` and specify
the desired values for the `name` and `adversarial_key` keywords in the `attack` module of a scenario configuration.
Valid values for each keyword are given in the table below.

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

### Preloaded Image Datasets
|             `name`             |        `adversarial_key`       |                Description                |               Attack               | Source Split |      x_shape     | x_type | y_shape | y_type |      Size      |
|:------------------------------:|:------------------------------:|:-----------------------------------------:|:----------------------------------:|:------------:|:----------------:|:------:|:-------:|:------:|:--------------:|
| "imagenet_adversarial"         | "adversarial"                  | ILSVRC12 adversarial image dataset for ResNet50  | Targeted, universal perturbation   |     test         | (nb, 224, 224, 3) |uint8   | (N,)    | int64  | 1000 images    |
| "resisc45_adversarial_224x224" |     "adversarial_univpatch"    | REmote Sensing Image Scene Classification |      Targeted, universal patch     |     test     | (nb, 224, 224, 3) |  uint8 |   (N,)  |  int64 | 5 images/class |
| "resisc45_adversarial_224x224" | "adversarial_univperturbation" | REmote Sensing Image Scene Classification | Untargeted, universal perturbation |     test     | (nb, 224, 224, 3) |  uint8 |   (N,)  |  int64 | 5 images/class |
| "apricot_dev_adversarial"      | ["adversarial", frcnn", "ssd", "retinanet"]                   | [Physical Adversarial Attacks on Object Detection](https://arxiv.org/abs/1912.08166)| Targeted, universal patch    | dev          | (nb, variable_height, variable_width, 3) | uint8 | n/a | dict | 138 images |
| "apricot_test_adversarial"     | ["adversarial", frcnn", "ssd", "retinanet"]                   | [Physical Adversarial Attacks on Object Detection](https://arxiv.org/abs/1912.08166)| Targeted, universal patch    | test          | (nb, variable_height, variable_width, 3) | uint8 | n/a | dict | 873 images |

##### APRICOT
Note: the APRICOT dataset contains splits for ["frcnn", "ssd", "retinanet"] rather than adversarial keys. See example below.
The APRICOT dataset contains labels and bounding boxes for both COCO objects and physical adversarial patches.
The label used to signify the patch is the `ADV_PATCH_MAGIC_NUMBER_LABEL_ID` defined in
[armory/data/adversarial_datasets.py](../armory/data/adversarial_datasets.py). Each image contains one adversarial
patch and a varying number of COCO objects (in some cases zero). COCO object class labels are one-indexed (start from 1)
in Armory <= 0.13.1 and zero-indexed in Armory > 0.13.1.

```json
"attack": {
    "knowledge": "white",
    "kwargs": {
        "batch_size": 1,
        "split": "frcnn"
    },
    "module": "armory.data.adversarial_datasets",
    "name": "apricot_dev_adversarial",
    "type": "preloaded",
```


### Preloaded Audio Datasets
|           `name`          | `adversarial_key`              |                     Description                    |               Attack               | Source Split |  x_shape  | x_type | y_shape | y_type | sampling_rate |      Size      |
|:-------------------------:|:-----------------:|:--------------------------------------------------:|:----------------------------------:|:------------:|:---------:|:------:|:-------:|:------:|:-------------:|:--------------:|
| "librispeech_adversarial" | "adversarial_perturbation      | Librispeech dev dataset for speaker identification | Targeted, universal perturbation   |     test     | (N, variable_length) |  int64 |   (N,)  |  int64 |    16 kHz     | ~5 sec/speaker |
| "librispeech_adversarial" | "adversarial_univperturbation" | Librispeech dev dataset for speaker identification | Untargeted, universal perturbation |     test     | (N, variable_length) |  int64 |   (N,)  |  int64 |    16 kHz     | ~5 sec/speaker |

### Preloaded Video Datasets
|            `name`            |      `adversarial_key`     |         Description        |               Attack               | Source Split |              x_shape              | x_type | y_shape | y_type |      Size      |
|:----------------------------:|:--------------------------:|:--------------------------:|:----------------------------------:|:------------:|:---------------------------------:|:------:|:-------:|:------:|:--------------:|
| "ucf101_adversarial_112x112" |     "adversarial_patch"    | UCF 101 Action Recognition | Untargeted, universal perturbation |     test     | (N, variable_frames, 112, 112, 3) |  uint8 |   (N,)  |  int64 | 5 videos/class |
| "ucf101_adversarial_112x112" | "adversarial_perturbation" | UCF 101 Action Recognition | Untargeted, universal perturbation          |     test     | (N, variable_frames, 112, 112, 3) |  uint8 |   (N,)  |  int64 | 5 videos/class |

### Preloaded Poison Datasets
|             `name`             |        `split_type`       |                Description                |               Attack               | Source Split |      x_shape     | x_type  | y_shape | y_type |      Size      |
|:------------------------------:|:------------------------------:|:-----------------------------------------:|:----------------------------------:|:------------:|:----------------:|:------:|:-------:|:------:|:--------------:|
| "gtsrb_poison"                 | poison                           | German Traffic Sign Poison Dataset        | Data poisoning                     |       train       |  (N, 48, 48, 3)  | float32 | (N,)    | int64  | 2220 images    |
| "gtsrb_poison"                 | poison_test                           | German Traffic Sign Poison Dataset        | Data poisoning                     |       test       |  (N, 48, 48, 3)  | float32 | (N,)    | int64  | 750 images    |
