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
![imagenette example](images/imagenette_example.png)

### TFDS Format: No Preprocessing
```
{'label': 9}
```
- excluding `image` key
- example shape: `[422, 500, 3]`

## `ucf101`
- config:
- split keys: `["train", "test"]`

### Example
![ucf101 example](images/ucf101_example.png)

### TFDS Format: No Preprocessing
```
{'label': 12}
```
- excluding `video` key
- example shape: `[118, 240, 320, 3]`

## `german_traffic_sign`
- config: `None`
- split keys: `["train", "test"]`

### Example
![german_traffic_sign example](images/german_traffic_sign_example.png)

### TFDS Format: No Preprocessing
```
{'filename': b'01952.bmp', 'label': 5}
```
- excluding `image` key
- example shape: `[40, 40, 3]`

## `carla_obj_det_dev`
- config: `None`
- split keys: `["dev"]`

## `carla_obj_det_train`
- config: `None`
- split keys: `["train", "val"]`

## `carla_obj_det_test`
- config: `None`
- split keys: `["test"]`

## `carla_over_obj_det_dev`
- config: `None`
- split keys: `["dev"]`

## `carla_over_obj_det_train`
- config: `None`
- split keys: `["train", "val"]`

## `apricot_dev`
- config: `None`
- split keys: `["retinanet", "frcnn", "ssd"]`

## `apricot_test`
- config: `None`
- split keys: `["retinanet", "frcnn", "ssd"]`

## `dapricot_dev`
Resolving with `load` errors

## `dapricot_test`
Resolving with `load` errors

## `coco`
- config: `["2014", "2017", "2017_panoptic"]`
- split keys: `["train", "test", "validation"]`
- `"test"` does not contain `"label"`
### Example
![coco example](images/coco_example.png)
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
- example shape: `[427, 640, 3]`
- `bbox` format: `[ymin, xmin, ymax, xmax]` normalized

### Armory Expected Input Format

#### Model

#### Attack

### Armory Output Format

## Video
