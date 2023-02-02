# TFDSv4 Dataset Migration Documentation

This file will act as a placeholder for any documentation related to the changes being made during the migration to [tfdsv4](https://github.com/twosixlabs/armory/milestone/18) with the hope that the discrepency between input/output shapes across all dataloaders & scenarios is clearer.

## High Level Changes

 Introduction of a `Task` which represents a group of datasets & their related scenarios. For example, the `Video` task would represent `carla_video_tracking` along with `carla_mot` and their respective dataloaders with the expectation both data loaders will produce output of the same shape.

 More to come...

## Script
The script below is intended to get information about the `tfds` format of a dataset prior to any calling any preprocessing functions defined in Armory to pass to a task/scenario. This script should work as long as the dataset has been built with `tfds` and uploaded to S3.
```python
from armory.datasets.load import load
import tensorflow_datasets as tfds

def get_ds_iterator(ds_dict, split):
    ds = ds_dict[split]
    iterator = iter(ds)
    return iterator

name = "imagenette"
config = None

info, ds_dict = load(name = name, config = config)
print(ds_dict.keys()) # dict_keys(['train', 'validation'])

iterator = get_ds_iterator(ds_dict, "validation")
x = iterator.__next__()
print(type(x), x.keys()) # dict, dict_keys(['image', 'label'])

print(x["image"].shape) # [422, 500, 3]

print(tfds.as_numpy({k:v for k,v in x.items() if k != "image"})) # {'label': 9}
```

## `imagenette`
- config: 
- split keys: `["train", "validation"]`
### Example
<!-- ![imagenette example](images/imagenette_example.png) -->
<img src="images/imagenette_example.png" alt="imagenette example" width="400"/>

### TFDS Format: No Preprocessing
```
{'label': 9}
```
- excluding `image` key
- example `shape`: `[422, 500, 3]`
- example `(min`, `max)`: `(0, 255)`

## `ucf101`
- config:
- split keys: `["train", "test"]`

### Example
<!-- ![ucf101 example](images/ucf101_example.png) -->
<img src="images/ucf101_example.png" alt="ucf101 example" width="400"/>

- single frame

### TFDS Format: No Preprocessing
```
{'label': 12}
```
- excluding `video` key
- example `shape`: `[118, 240, 320, 3]`
- example `(min`, `max)`: `(0, 255)`

## `german_traffic_sign`
- config: `None`
- split keys: `["train", "test"]`

### Example
<!-- ![german_traffic_sign example](images/german_traffic_sign_example.png) -->
<img src="images/german_traffic_sign_example.png" alt="german_traffic_sign example" width="200"/>

### TFDS Format: No Preprocessing
```
{'filename': b'01952.bmp', 'label': 5}
```
- excluding `image` key
- example `shape`: `[40, 40, 3]`
- example `(min`, `max)`: `(2, 247)`

## `so2sat`
- config: `None`
- split keys: `["train", "validation"]`

### Example
<!-- ![so2sat example](images/so2sat_example.png) -->
<img src="images/so2sat_example.png" alt="so2sat example" width="200"/>

### TFDS Format: No Preprocessing
```
{'label': 7, 'sample_id': 2308}
```
- excluding `image` key
- example `shape`: `[32, 32, 3]`
- example `(min`, `max)`: `(83, 255)`

## `carla_obj_det_dev`
- config: `None`
- split keys: `["dev"]`

### Example
<!-- ![carla_obj_det_dev_rgb example](images/carla_obj_det_dev_rgb_example.png) -->
<img src="images/carla_obj_det_dev_rgb_example.png" alt="carla_obj_det_dev_rgb example" width="400"/>
<img src="images/carla_obj_det_dev_depth_example.png" alt="carla_obj_det_dev_depth example" width="400"/>
<img src="images/carla_obj_det_dev_mask_example.png" alt="carla_obj_det_dev_mask example" width="400"/>

- RGB, depth, and `mask`
- boxes were drawn

### TFDS Format: No Preprocessing
```
{'categories': {'id': array([1, 2, 3, 4]),
  'name': array([b'Pedestrian', b'Vehicle', b'TrafficLight', b'Patch'], dtype=object),
  'supercategory': array([b'Pedestrian', b'Vehicle', b'TrafficLight', b'Patch'], dtype=object)},
 'images': {'file_name': array([b'_out/sensor.camera.rgb.1/config_obj_det_security_camera_perspective_1_00000612.png', b'_out/sensor.camera.depth.2/config_obj_det_security_camera_perspective_1_00000612.png'], dtype=object),
  'height': array([960, 960]),
  'id': array([84190894, 84190894]),
  'width': array([1280, 1280])},
 'objects': {'area': array([  936,   385,   433,   602,   313,  1930,   558,  1074, 54479, 11805]),
  'boxes': array([[0.48333332, 0.7164062 , 0.54895836, 0.74609375],
         [0.4875    , 0.71171874, 0.5385417 , 0.734375  ],
         [0.48645833, 0.60234374, 0.5395833 , 0.615625  ],
         ...
         [0.44479167, 0.        , 0.66770834, 0.28828126],
         [0.49791667, 0.2703125 , 0.60625   , 0.40859374]], dtype=float32),
  'id': array([ 97,  98,  99, 100, 101, 102, 103, 105, 106, 109]),
  'image_id': array([84190894, 84190894, 84190894, 84190894, 84190894, 84190894, 84190894, 84190894, 84190894, 84190894]),
  'is_crowd': array([False,  True, False, False,  True, False, False, False, False, False]),
  'labels': array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2])},
 'patch_metadata': {'avg_patch_depth': 0.47842735052108765,
  'gs_coords': array([[792, 445],
         [890, 442],
         [890, 507],
         [792, 493]], dtype=int32)}
}
```
- excluding `image` key
- excluding `mask` key from `patch_metadata`
- value for `boxes` key is truncated in `objects`
- example `shape`:
  - `image`: `[2, 960, 1280, 3]`
  - `mask`: `[960, 1280, 3]`
- example `(min`, `max)`:
  - `image`: `(0, 255)`
  - `mask`: `(0, 255)`
- `boxes` format in `objects`: `[ymin, xmin, ymax, xmax]` normalized

## `carla_obj_det_train`
- config: `None`
- split keys: `["train", "val"]`

### Example
<!-- ![carla_obj_det_dev_rgb example](images/carla_obj_det_dev_rgb_example.png) -->
<img src="images/carla_obj_det_train_rgb_example.png" alt="carla_obj_det_train_rgb example" width="400"/>
<img src="images/carla_obj_det_train_depth_example.png" alt="carla_obj_det_train_depth example" width="400"/>

- RGB, depth
- boxes were drawn

### TFDS Format: No Preprocessing
```
{'categories': {'id': array([1, 2, 3]),
  'name': array([b'Pedestrian', b'Vehicle', b'TrafficLight'], dtype=object),
  'supercategory': array([b'Pedestrian', b'Vehicle', b'TrafficLight'], dtype=object)},
 'images': {'file_name': array([b'config4.00002910.png', b'config4.00002910.png'], dtype=object),
  'height': array([960, 960]),
  'id': array([3478, 3478]),
  'width': array([1280, 1280])},
 'objects': {'area': array([ 3474,   694,   114,   206,   122, 11360,   127,   219]),
  'boxes': array([[0.49791667, 0.15390626, 0.5385417 , 0.26875   ],
         [0.50208336, 0.621875  , 0.52916664, 0.6484375 ],
         [0.49583334, 0.5210937 , 0.5072917 , 0.5445312 ],
         [0.496875  , 0.5992187 , 0.5104167 , 0.61640626],
         [0.5052083 , 0.71953124, 0.51979166, 0.74296874],
         [0.45833334, 0.        , 0.56666666, 0.12734374],
         [0.484375  , 0.2484375 , 0.5104167 , 0.25546876],
         [0.496875  , 0.75625   , 0.53020835, 0.765625  ]], dtype=float32),
  'id': array([40379, 40380, 40381, 40382, 40383, 40384, 40385, 40386]),
  'image_id': array([3478, 3478, 3478, 3478, 3478, 3478, 3478, 3478]),
  'is_crowd': array([False, False, False, False, False, False, False, False]),
  'labels': array([2, 2, 2, 2, 2, 2, 1, 1])}}
```
- excluding `image` key
- ***NO*** `patch_metadata` key
- example `shape`: `[2, 960, 1280, 3]`
- example `(min`, `max)`: `(0, 255)`
- `boxes` format in `objects`: `[ymin, xmin, ymax, xmax]` normalized

## `carla_obj_det_test`
- config: `None`
- split keys: `["test"]`

## `carla_over_obj_det_dev`
- config: `None`
- split keys: `["dev"]`

### Example
<!-- ![carla_obj_det_dev_rgb example](images/carla_obj_det_dev_rgb_example.png) -->
<img src="images/carla_over_obj_det_dev_rgb_example.png" alt="carla_over_obj_det_dev_rgb example" width="400"/>
<img src="images/carla_over_obj_det_dev_depth_example.png" alt="carla_over_obj_det_dev_depth example" width="400"/>
<img src="images/carla_over_obj_det_dev_mask_example.png" alt="carla_over_obj_det_dev_mask example" width="400"/>

- RGB, depth, and `mask`
- boxes were drawn

### TFDS Format: No Preprocessing
```
{'categories': {'id': array([1, 2]),
  'name': array([b'Pedestrian', b'Vehicle'], dtype=object),
  'supercategory': array([b'Pedestrian', b'Vehicle'], dtype=object)},
 'images': {'file_name': array([b'00002040.png', b'depth/00002040.png'], dtype=object),
  'height': array([960, 960]),
  'id': array([69734696, 69734696]),
  'width': array([1280, 1280])},
 'objects': {'area': array([1869, 3543,  120,  107,  145,  102,  116,  130,  139,  149,  122,
          154, 3893, 7951,  172]),
  'boxes': array([[0.69375   , 0.52734375, 0.775     , 0.553125  ],
         [0.65833336, 0.48359376, 0.7583333 , 0.51640624],
         [0.44791666, 0.13359375, 0.45625   , 0.15234375],
         ...
         [0.59791666, 0.38359374, 0.75208336, 0.4296875 ],
         [0.953125  , 0.45      , 0.9947917 , 0.46015626]], dtype=float32),
  'id': array([468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
         481, 482]),
  'image_id': array([69734696, 69734696, 69734696, 69734696, 69734696, 69734696,
         69734696, 69734696, 69734696, 69734696, 69734696, 69734696,
         69734696, 69734696, 69734696]),
  'is_crowd': array([False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False]),
  'labels': array([2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])},
 'patch_metadata': {'avg_patch_depth': 40.0118408203125,
  'gs_coords': array([[438, 593],
         [669, 591],
         [672, 861],
         [442, 863]], dtype=int32)}
}
```
- excluding `image` key
- excluding `mask` key from `patch_metadata`
- value for `boxes` key is truncated in `objects`
- example `shape`:
  - `image`: `[2, 960, 1280, 3]`
  - `mask`: `[960, 1280, 3]`
- example `(min`, `max)`:
  - `image`: `(0, 255)`
  - `mask`: `(0, 255)`
- `boxes` format in `objects`: `[ymin, xmin, ymax, xmax]` normalized

## `carla_over_obj_det_train`
- config: `None`
- split keys: `["train", "val"]`

### Example
<!-- ![carla_over_obj_det_train_rgb example](images/carla_over_obj_det_train_rgb_example.png) -->
<img src="images/carla_over_obj_det_train_rgb_example.png" alt="carla_over_obj_det_train_rgb example" width="400"/>
<img src="images/carla_over_obj_det_train_depth_example.png" alt="carla_over_obj_det_train_depth example" width="400"/>

- RGB, depth
- boxes were drawn

### TFDS Format: No Preprocessing
```
{'categories': {'id': array([1, 2]),
  'name': array([b'Pedestrian', b'Vehicle'], dtype=object),
  'supercategory': array([b'Pedestrian', b'Vehicle'], dtype=object)},
 'images': {'file_name': array([b'route1_20z_310deg.00000801.png',
         b'route1_20z_310deg.00000801.png'], dtype=object),
  'height': array([960, 960]),
  'id': array([1080, 1080]),
  'width': array([1280, 1280])},
 'objects': {'area': array([ 312,  169,  203,  431,  914,  165,  132,  285,  146,  180,  154,
          195,  203,  149,  171,  149,  875,  178,  332,  257,  129,  199,
          336,  270,  222, 6810, 2756, 3518, 2798, 4235,  154,  194, 2831,
          740, 4291,  256,  151]),
  'boxes': array([[0.5864583 , 0.603125  , 0.61875   , 0.6226562 ],
         [0.17916666, 0.9828125 , 0.20625   , 1.        ],
         [0.28020832, 0.6890625 , 0.31666666, 0.703125  ],
         ...
         [0.28333333, 0.92265624, 0.31145832, 0.9476563 ],
         [0.29375   , 0.9140625 , 0.31458333, 0.9453125 ]], dtype=float32),
  'id': array([23278, 23279, 23280, 23281, 23282, 23283, 23284, 23285, 23286,
         23287, 23288, 23289, 23290, 23291, 23292, 23293, 23294, 23295,
         23296, 23297, 23298, 23299, 23300, 23301, 23302, 23303, 23304,
         23305, 23306, 23307, 23308, 23309, 23310, 23311, 23312, 23313,
         23314]),
  'image_id': array([1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080,
         1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080,
         1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080,
         1080, 1080, 1080, 1080]),
  'is_crowd': array([False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False]),
  'labels': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2])}}
```
- excluding `image` key
- ***NO*** `patch_metadata` key
- value for `boxes` key is truncated in `objects`
- example `shape`: `[2, 960, 1280, 3]`
- example `(min`, `max)`: `(0, 255)`
- `boxes` format in `objects`: `[ymin, xmin, ymax, xmax]` normalized

## `apricot_dev`
- config: `None`
- split keys: `["retinanet", "frcnn", "ssd"]`

## `apricot_test`
- config: `None`
- split keys: `["retinanet", "frcnn", "ssd"]`

### Example
<img src="images/apricot_test_frcnn_example.png" alt="apricot_test_frcnn example" width="400"/>

- boxes were drawn

### TFDS Format: No Preprocessing
```
{'objects': {'angle': array([[-1, -1, -1],
         [-1, -1, -1],
         [-1, -1, -1],
         [-1, -1, -1],
         [ 0,  1,  0]]),
  'area': array([ 124740,  293600,  736291, 3161754,  206435]),
  'boxes': array([[0.4017857 , 0.        , 0.4799107 , 0.13095239],
         [0.8177083 , 0.8673942 , 0.999752  , 0.9996693 ],
         [0.8492063 , 0.12466931, 0.999752  , 0.5257937 ],
         [0.6113591 , 0.2602513 , 0.999752  , 0.92791003],
         [0.5990501 , 0.6170437 , 0.71292907, 0.76571757]], dtype=float32),
  'id': array([         854,          855,          856,          857,
         910000000039]),
  'image_id': array([600039, 600039, 600039, 600039, 600039]),
  'is_crowd': array([False, False, False, False, False]),
  'is_warped': array([False, False, False, False, False]),
  'labels': array([ 2, 62, 62, 67, 12]),
  'patch_id': array([-1, -1, -1, -1,  4])}}
```
- ***ONLY*** included `objects` key
- example `shape`: `[4032, 3024, 3]`
- example `(min`, `max)`: `(0, 255)`
- `boxes` format in `objects`: `[ymin, xmin, ymax, xmax]` normalized

## `dapricot_dev`
Need to resolve `load` errors

## `dapricot_test`
Need to resolve `load` errors

## `cifar10`
- config: `None`
- split keys: `["train", "test"]`

### Example
<!-- ![cifar10 example](images/cifar10_example.png) -->
<img src="images/cifar10_example.png" alt="cifar10 example" width="200"/>

### TFDS Format: No Preprocessing
```
{'id': b'test_09933', 'label': 7}
```
- excluding `image` key
- example `shape`: `[32, 32, 3]`
- example `(min`, `max)`: `(5, 255)`

## `cifar100`
- config: `None`
- split keys: `["train", "test"]`

### Example
<!-- ![cifar100 example](images/cifar100_example.png) -->
<img src="images/cifar100_example.png" alt="cifar100 example" width="200"/>

### TFDS Format: No Preprocessing
```
{'coarse_label': 8, 'id': b'test_09933', 'label': 43}
```
- excluding `image` key
- example `shape`: `[32, 32, 3]`
- example `(min`, `max)`: `(9, 234)`

## `digit`
- config: `None`
- split keys: `["train", "test"]`

### Example
![digit example](images/digit_example.png)

### TFDS Format: No Preprocessing
```
{'filename': b'5_theo_2.wav', 'label': 5, 'user': b'theo'}
```
- excluding `audio` key
- example `shape`: `[2139]`
- example `(min`, `max)`: `(-874, 1262)`

## `xview`
- config: `None`
- split keys: `["train", "test"]`

### Example
<!-- ![xview example](images/xview_example.png) -->
<img src="images/xview_example.png" alt="xview example" width="400"/>

- boxes were drawn

### TFDS Format: No Preprocessing
```
{'objects': {'area': array([  99,  271,  124,  475, 4523, 1440,  623,  753,  775,  701,  252,
         1080,  837, 1130,  864,  414,  493,  858,  243,  898,  352, 1023,
         1121,  758,  450,  703,  492,  679,  544,  800]),
  'boxes': array([[0.955, 0.945, 1.   , 1.   ],
         [0.215, 0.44 , 0.3  , 0.52 ],
         [0.975, 0.605, 1.   , 0.73 ],
         ...
         [0.435, 0.   , 0.595, 0.085],
         [0.515, 0.69 , 0.715, 0.79 ]], dtype=float32),
  'id': array([439394, 439395, 439396, 439397, 439398, 439399, 439400, 439401,
         439402, 439403, 439404, 439405, 439406, 439407, 439408, 439409,
         439410, 439411, 439412, 439413, 439414, 439415, 439416, 439417,
         439418, 439419, 439420, 439421, 439422, 439423]),
  'image_id': array([16932, 16932, 16932, 16932, 16932, 16932, 16932, 16932, 16932,
         16932, 16932, 16932, 16932, 16932, 16932, 16932, 16932, 16932,
         16932, 16932, 16932, 16932, 16932, 16932, 16932, 16932, 16932,
         16932, 16932, 16932]),
  'is_crowd': array([False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False]),
  'labels': array([49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49,
         49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49])}}
```
- excluding `image` key
- value for `boxes` key is truncated in `objects`
- example `shape`: `[200, 200, 3]`
- example `(min`, `max)`: `(0, 255)`
- `boxes` format in `objects`: `[ymin, xmin, ymax, xmax]` normalized

## `resisc45`
- config: `None`
- split keys: `["train", "validation", "test"]`

### Example
<!-- ![resisc45 example](images/resisc45_example.png) -->
<img src="images/resisc45_example.png" alt="resisc45 example" width="200"/>

### TFDS Format: No Preprocessing
```
{'filename': b'sea_ice_639.jpg', 'label': 35}
```
- excluding `image` key
- example `shape`: `[256, 256, 3]`
- example `(min`, `max)`: `(51, 217)`

## `mnist`
- config: `None`
- split keys: `["test", "train"]`

### Example
<!-- ![mnist example](images/mnist_example.png) -->
<img src="images/mnist_example.png" alt="mnist example" width="200"/>

### TFDS Format: No Preprocessing
```
{'label': 2}
```
- excluding `image` key
- example `shape`: `[28, 28, 1]`
- example `(min`, `max)`: `(0, 255)`

## `coco`
- config: `["2014", "2017", "2017_panoptic"]`
- split keys: `["train", "test", "validation"]`
- `"test"` does not contain `"label"`

### Example
<!-- ![coco example](images/coco_example.png) -->
<img src="images/coco_example.png" alt="coco example" width="400"/>

- boxes were drawn

### Original Format
```
[{'area': 49565.45300000001,
  'iscrowd': 0,
  'image_id': 133418,
  'bbox': [303.54, 36.62, 193.68, 382.83],
  'category_id': 1,
  'id': 444415},
 {'area': 13487.548600000002,
  'iscrowd': 0,
  'image_id': 133418,
  'bbox': [386.64, 267.68, 253.36, 159.32],
  'category_id': 43,
  'id': 657440}]
```
- excluding `segmentation` key
- `bbox` format: `[xmin, ymin, width, height]`

### TFDS Format: No Preprocessing
```
{'image/filename': b'000000133418.jpg',
 'image/id': 133418,
 'objects': {'area': array([49565, 13487]),
  'bbox': array([[0.08576112, 0.47428125, 0.9823185 , 0.77690625],
         [0.62688524, 0.604125  , 1.        , 1.        ]], dtype=float32),
  'id': array([444415, 657440]),
  'is_crowd': array([False, False]),
  'label': array([ 0, 38])}}
```
- excluding `image` key
- example `shape`: `[427, 640, 3]`
- example `(min`, `max)`: `(0, 255)`
- `bbox` format in `objects`: `[ymin, xmin, ymax, xmax]` normalized

### Armory Expected Input Format

#### Model

#### Attack

### Armory Output Format

## Video
