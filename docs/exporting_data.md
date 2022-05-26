# Exporting Data


### What do we mean by exporting data?
Armory provides the capability of saving off benign and adversarial examples
to the Armory output directory. The table below indicates what kind of output
you can expect, based on the scenario:

| Scenario               | Output File Type(s) | 
|------------------------|---------------------|
| audio_asr              | .wav                | 
| audio_classification   | .wav                |          | 
| carla_video_tracking   | .png, .mp4          | 
| dapricot_scenario      | .png                | 
| image_classification   | .png                | 
| object_detection       | .png, .json         | 
| video_ucf101_scenario  | .png, .mp4          | 

Armory also outputs a pickle file containing ground-truth labels, benign predictions, and adversarial predictions.

### How to Export Data
In the `"scenario"`portion of the config, set the `"export_batches"` field to the
number of batches of data you'd like exported (see [configuration_files.md](configuration_files.md#exporting-data). You can also set this field to `true` if 
you'd like for all batches to be exported. If you do not want to export data, you do 
not need to include this field in the config. 

Please see the following example, in which one batch of data will be exported:

```commandline
  "scenario": {
        "kwargs": {},
        "module": "armory.scenarios.image_classification",
        "name": "ImageClassificationTask",
        "export_batches": 1
    },

```


Note: If using Armory < 0.15.0, please instead use the `"export_samples"` field in the `"scenario"` config. This field was deprecated in Armory 0.15.0.


### Exporting/Viewing Data Interactively
If you are running Armory with the `--interactive` flag, you can interactively view and/or save off data examples. 
Please see [docker.md](docker.md#interactive-use) for instructions on how to run Armory interactively and [running_armory_scenarios_interactively.ipynb](../notebooks/running_armory_scenarios_interactively.ipynb) for a tutorial which includes instructions for exporting samples. Once you've attached
to the container, please see the following code snippets for an example of how to view and/or save off data examples:

First, we'll simply load our scenario config and evaluate on one batch of data:
```commandline
>>> from armory.scenarios.main import get as get_scenario
>>> s = get_scenario("/armory/tmp/2022-03-18T163008.249437/interactive-config.json").load()  # load config
>>> s.next()  # load batch of data
>>> s.evaluate_current()  # make benign prediction, generate adversarial sample, make adversarial prediction
```

Now at this point, let's say you'd like to save off the benign and adversarial example:
```commandline
>>> s.x.shape
(1, 32, 32, 3)
>>> s.sample_exporter.export(s.x[0], "benign_x")
>>> s.sample_exporter.export(s.x_adv[0], "adversarial_x")
```

After calling this method, the images are saved to the scenario output directory:
```commandline
ls ~/.armory/outputs/2022-03-18T163008.249437/saved_samples/
adversarial_x.png  benign_x.png
```

If, instead of writing to disk, you'd like to return an individual sample, use the `get_sample()` method:
```commandline
>>> img = s.sample_exporter.get_sample(s.x_adv[0])
>>> type(img)
<class 'PIL.Image.Image'>
```

### Exporting Data With Bounding Boxes
For object detection and video tracking scenarios, Armory exports a set of raw images as well as images containing ground-truth (red) 
and predicted (white) bounding boxes. This will all occur automatically if using Armory in the normal/non-interactive mode, but we include
the following interactive example as well:

In the example below, we've already loaded an xView object detection scenario and run `evaluate_current()` for a single batch before running the following to export
the adversarial image with ground-truth and predicted boxes overlaid:
```commandline
>>> s.sample_exporter.export(
        s.x_adv[0], 
        "adversarial_x_with_boxes", 
        y=s.y[0], 
        y_pred=s.y_pred_adv[0], 
        with_boxes=True)
```


```commandline
ls ~/.armory/outputs/2022-03-18T181736.925088/saved_samples/
adversarial_x_with_boxes.png
```

As depicted earlier, the `get_sample()` method can be used to return the PIL image. The boolean `with_boxes` kwarg can be used to add
bounding boxes to the image. When this is set to `True`, you must provide values for at least one of `y` and `y_pred`.
```commandline
>>> adv_img = s.sample_exporter.get_sample(s.x_adv[0])
>>> type(adv_img)
<class 'PIL.Image.Image'>

>>> adv_img_with_boxes = s.sample_exporter.get_sample(s.x_adv[0], with_boxes=True, y=s.y[0], y_pred=s.y_pred_adv[0])
>>> type(adv_img_with_boxes)
<class 'PIL.Image.Image'>

>>> s.sample_exporter.get_sample(s.x[0], with_boxes=True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/workspace/armory/utils/export.py", line 205, in get_sample
    raise TypeError("Both y and y_pred are None, but with_boxes is True")
TypeError: Both y and y_pred are None, but with_boxes is True

```
If you'd like to include only ground-truth boxes (or only predicted boxes), provide an arg for only `y` (or only `y_pred`).


### Exporting Multimodal Data
#### Multimodal CARLA Object Detection
For the multimodal CARLA scenario, depth images are automatically outputted in addition to RGB when `"export_batches"` is set in the config. If you'd like to interactively return the depth image, 
call `get_sample(x_i[..., 3:])`.

#### So2Sat Image Classification
The `get_sample()` method for the So2Sat scenario exporter takes a `modality` arg which must be one of `{'vh', 'vv', 'eo'}`. Calling `export()` will save off all three types of examples, which will occur automatically when running a config where `"export_batches"` is set.

### Exporting COCO-Formatted Bounding Boxes
For object detection scenarios, if `"export_batches"` is set in the config file, Armory will output COCO-formatted JSON
files with ground-truth and predicted bounding boxes. These are saved in the scenario output directory as 
`ground_truth_boxes_coco_format.json`, `benign_predicted_boxes_coco_format.json`, and `adversarial_predicted_boxes_coco_format.json`. Note that
this functionality exists separately from the exporters described in this document.
