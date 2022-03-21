# Exporting Data


### What do we mean by exporting data?
Armory provides the capability of saving off benign and adversarial examples
to the Armory output directory. The table below indicates what kind of output
you can expect, based on the scenario:

| Scenario               | Output File Type(s) | 
|------------------------|---------------------|
| audio_asr              | .wav                | 
| audio_classification   | .wav                | 
| carla_object_detection | .png                | 
| carla_video_tracking   | .png, .mp4          | 
| dapricot_scenario      | .png                | 
| image_classification   | .png                | 
| object_detection       | .png                | 
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
Please see [docker.md](docker.md#interactive-use) for instructions on how to run Armory interactively. Once you've attached
to the container, please see the following code snippet for an example of how to view and/or save off data examples:

```commandline
>>> from armory import scenarios
>>> s = scenarios.get("/armory/tmp/2022-03-18T163008.249437/interactive-config.json").load()  # load cifar10 config
>>> s.next()  # load batch of data
>>> s.evaluate_current()  # make benign prediction, generate adversarial sample, make adversarial prediction
```

Now at this point, let's say you'd like to save off the benign and adversarial example:
```commandline
>>> s.x.shape
(1, 32, 32, 3)
>>> s.sample_exporter.export(x=s.x, x_adv=s.x_adv)
```

After calling this method, the images are saved to the scenario output directory:
```commandline
ls ~/.armory/outputs/2022-03-18T163008.249437/saved_samples/
0_adversarial.png  0_benign.png
```

If, instead of writing to disk, you'd like to return an individual sample, use the `get_sample()` method:
```commandline
>>> img = s.sample_exporter.get_sample(s.x_adv[0])
>>> type(img)
<class 'PIL.Image.Image'>
```

Note that the `export()` method expects a batch of data, e.g. an image of shape `(nb, H, W, C)`, whereas `get_sample()` expects 
an individual data example.

### Exporting Data With Bounding Boxes
For object detection and video tracking scenarios, Armory exports a set of raw images as well as images containing ground-truth (red) 
and predicted (white) bounding boxes. This will all occur automatically if using Armory in the normal/non-interactive mode, but we include
the following interactive example as well:

In the example below, we've already loaded an xView object detection scenario and run `evaluate_current()` for a single batch before running the following:
```commandline
>>> s.sample_exporter.export(
                      x=s.x, 
                      x_adv=s.x_adv, 
                      y=s.y, 
                      y_pred_clean=s.y_pred, 
                      y_pred_adv=s.y_pred_adv,
                      plot_boxes=True)
```

The call above yields the following output:

```commandline
ls ~/.armory/outputs/2022-03-18T181736.925088/saved_samples/
0_adversarial.png  0_adversarial_with_boxes.png  0_benign.png  0_benign_with_boxes.png
```

You could also export only the raw images, absent boxes, with the following call:
```commandline
>>> s.sample_exporter.export(x=s.x, x_adv=s.x_adv)
```

As depicted earlier, the `get_sample()` method can be used to return the PIL image. In addition, the `get_sample_with_boxes()` method can be used to obtain the 
PIL image including bounding boxes:
```commandline
>>> adv_img = s.sample_exporter.get_sample(s.x_adv[0])
>>> type(adv_img)
<class 'PIL.Image.Image'>

>>> adv_img_with_boxes = s.sample_exporter.get_sample_with_boxes(s.x_adv[0], y_i=s.y[0], y_i_pred=s.y_pred_adv[0])
>>> type(adv_img_with_boxes)
<class 'PIL.Image.Image'>
```
If you'd only like to include ground-truth boxes (or only predicted boxes), don't provide an arg for `y_i` or `y_i_pred`.

### Exporting Multimodal Data
#### Multimodal CARLA Object Detection
For the multimodal CARLA scenario, depth images are outputted in addition to RGB. Depth images can be interactively obtained using the 
`get_depth_sample()` method.

#### So2Sat Image Classification
The So2Sat scenario exporter contains `get_vh_sample()`, `get_vv_sample()`, and `get_eo_samples()` methods, the last of which returns a list of 
PIL images. Calling `export()` will save off all three types of examples.

