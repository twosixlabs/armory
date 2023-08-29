# `armory utils` usage
Additional utility functions provided to support armory usage. Information available with `armory utils --help` as well.

## Provided utilities
1. [get-branch](#get-branch)
2. [rgb-convert](#rgb-convert)
3. [plot-mAP-by-giou](#plot-map-by-giou)
4. [shape-gen](#shape-gen)


## [get-branch](../armory/cli/tools.py#L24)
Requires `GitPython` to be installed in your current environment. Reports the current branch for armory, useful when dealing with multiple installs.
```
❯ armory utils get-branch
2023-08-15 20:08:56  2s INFO     armory.cli.tools:log_current_branch:29 Armory version: 0.16.6.post0+gf529d887.d20230615
2023-08-15 20:08:56  2s INFO     armory.cli.tools:log_current_branch:38 Git branch: master
```

## [rgb-convert](../armory/cli/tools.py#L49)
Requires `numpy`, `matplotlib`, and `Pillow` to be installed in your current environment.

Converts rgb depth images to and from `log` and `linear` formats.

```
usage: armory rgb-convert [-h] [-d] [--log-level LOG_LEVEL] [--headless] [--output OUTPUT] [--save] [--format {linear,log}] input [input ...]

positional arguments:
  input                 Path to depth image(s) to convert

optional arguments:
  --headless            Don't show converted depth image using matplotlib
                          Note: if using over ssh, must use --headless flag or must
                                have a GUI backend installed (ex. PyQt5) with X11 forwarding.
                          See:  https://matplotlib.org/faq/usage_faq.html#what-is-a-backend
  --output OUTPUT       Path to save converted depth image to
                          Note: Cannot be used with multiple input images.
  --save                Save converted depth image to <input_path>_<format>.png
  --format {linear,log}
                        Format used with --save or --output flag.
```
![demo.gif](https://github.com/jprokos26/armory/blob/external-links/rgb-convert-demo.gif?raw=true)

## [plot-mAP-by-giou](../armory/postprocessing/plot_patch_aware_carla_metric.py#L94)
Visualize the output of the metric `object_detection_AP_per_class_by_giou_from_patch`.

```
usage: armory plot-mAP-by-giou [-h] [--flavors {cumulative_by_max_giou,cumulative_by_min_giou,histogram_left} [{cumulative_by_max_giou,cumulative_by_min_giou,histogram_left} ...]] [--headless] [--output OUTPUT]
                               [--exclude-classes] [-d] [--log-level LOG_LEVEL]
                               input

positional arguments:
  input                 Path to json. Must have 'results.adversarial_object_detection_AP_per_class_by_giou_from_patch' key.

optional arguments:
  --flavors {cumulative_by_max_giou,cumulative_by_min_giou,histogram_left} [{cumulative_by_max_giou,cumulative_by_min_giou,histogram_left} ...]
                        Flavors of mAP by giou to plot. Subset of ['cumulative_by_max_giou', 'cumulative_by_min_giou', 'histogram_left'] or None to plot all.
  --headless            Don't show the plot
  --output OUTPUT       Path to save the plot
  --exclude-classes     Don't include subplot for each class.
```

See [our write-up](https://docs.google.com/document/d/1_8_nRQmHhK5ieHNcGhoRtBZcRY_cXid6e5keySER5eU/edit?usp=sharing) for more details.

## [shape-gen](../armory/utils/shape_gen.py)

    usage: armory shape-gen [-h] [--output-dir OUTPUT_DIR] [--show] {asterisk,circle,grid,concentric_circles,jxcr_gear,sierpinski,all}

    Generate shapes as png files

    Available shapes:
            asterisk
            circle
            grid
            concentric_circles
            jxcr_gear
            sierpinski
            all

    optional arguments:
      -h, --help            show this help message and exit
      --output-dir OUTPUT_DIR
      --show                Show the generated shape using matplotlib
