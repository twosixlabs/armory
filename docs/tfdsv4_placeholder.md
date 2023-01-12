# TFDSv4 Dataset Migration Documentation

This file will act as a placeholder for any documentation related to the changes being made during the migration to [tfdsv4](https://github.com/twosixlabs/armory/milestone/18) with the hope that the discrepency between input/output shapes across all dataloaders & scenarios is clearer.

## High Level Changes

 Introduction of a `Task` which represents a group of datasets & their related scenarios. For example, the `Video` task would represent `carla_video_tracking` along with `carla_mot` and their respective dataloaders with the expectation both data loaders will produce output of the same shape.

 More to come...

## COCO
### Example
![coco example](images/coco_example.png)

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
- `bbox` format: `[ymin, xmin, ymax, xmax]` normalized

### Armory Expected Input Format

#### Model

#### Attack

### Armory Output Format

## Video
