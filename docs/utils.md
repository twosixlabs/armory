# `armory utils` usage
Additional utility functions provided to support armory usage. Information available with `armory utils --help` as well.

## Provided utilities
1. [get-branch](#get-branch)
2. [rgb-convert](#rgb-convert)
3. [plot-mAP-by-giou](#plot-map-by-giou)
4. [shape-gen](#shape-gen)
5. [collect-outputs](#collect-outputs)


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


## [collect-outputs](../armory/cli/tools/collect_outputs.py)
Generate evaluation tables automatically using various flags. Optionally organize by performer, attack type, or any other config field.

Additionally supply `--clean` to automatically clean up your output directory. This:
- Removes folders with just `armory-log.txt` and `colored-log.txt`
- Removes pairs of duplicate png images from `saved_samples` where `...x.png` is equivalent to `...x_adv.png`.
- Removed files are placed in `ARMORY_OUTPUT_DIR/.cleaned`
_Note that extracted config path is always relative to CWD_

Currently only supports headers defined by HEADERS in [armory/cli/tools/collect_outputs.py#L376](../armory/cli/tools/collect_outputs.py#L376).


```shell
Convert runs from the output directory into tables

options:
  -h, --help            show this help message and exit
  --glob GLOB, -g GLOB  Glob pattern to match json outputs. Defaults to `*.json`.
  --output OUTPUT, -o OUTPUT
                        Path to output tables. Defaults to /home/jonathan.prokos/.armory/results/ATTACK.md where str format placeholder is replaced with ATTACK name if supplied.
  --clean               Clean up all failed runs (directories containing _only_ {armory,colored}-log.txt).
                        Moves them to a new directory called .cleaned.
  --unify [ATTACK ...]  Unify results from multiple attacks into a single markdown file. Takes a list of attack names to unify.
                        Defaults to all attacks if no attack is supplied. Does not output individual tables. 
  --collate [KWARG], -c [KWARG]
                        Combine attack results based on the supplied kwarg. Defaults to `config.metric.task`.
  --absolute            Use absolute path for hyperlinks in the output tables. Does not create symbolic links.
  --config-dir CONFIG_DIR
                        Path to link run configs from. Defaults to current directory.
  --default DEFAULT     Default attack to use for headers. Defaults to CARLAAdversarialPatchPyTorch.
  --sort [HEADER]       Sort results by the supplied header(s).
  --filter FILTER       Filter results to only those matching the supplied regex.
  --ignore-header IGNORE_HEADER [IGNORE_HEADER ...], -x IGNORE_HEADER [IGNORE_HEADER ...]
                        Ignore the supplied headers when parsing results.
  --ignore-kwargs IGNORE_KWARGS [IGNORE_KWARGS ...]
                        Ignore the supplied kwargs when parsing attack params.
  --keep-kwargs KEEP_KWARGS [KEEP_KWARGS ...]
                        Only keep the supplied kwargs when parsing attack params.
  --ignore-short-runs [IGNORE_SHORT_RUNS]
                        Ignore runs with runtime less than this value (in minutes). Defaults to 1 minute.
  -d, --debug           synonym for --log-level=armory:debug
  --log-level LOG_LEVEL
                        set log level per-module (ex. art:debug) can be used mulitple times
```

Example usage:
```shell
cd ~/git/twosixlabs/gard-evaluatsion/MITM_sleeper_agent/jprokos26
python3 -m armory utils collect-outputs --collate --glob "*Poison*.json" --sort --filter "diffusion"
cp -Lr ~/.armory/results/SleeperAgentAttack* .
git commit -am "Adding diffusion sleeper results"
```
Which gives me the following in a [markdown file](https://github.com/twosixlabs/gard-evaluations/blob/eval7-jp-MITM_sleeper/MITM_sleeper_agent/jprokos26/SleeperAgentAttack.md):

Run|Defense|Dataset|Attack|Attack Params|Poison %|Attack Success Rate|Accuracy (Benign/Poisoned)|output
---|---|---|---|---|---|---|---|---
[configs/diffusion/00673/cifar10_sleeper_agent_p10_MITM.json](configs/diffusion/00673/cifar10_sleeper_agent_p10_MITM.json)|BoostedWeakLearners|cifar10|SleeperAgentAttack|epsilon=0.0627 k_trigger=1000 lrs=[[0.1, 0.01, 0.001, 0.0001, 1e-05], [250, 350, 400, 430, 460]] max_epochs=500 max_trials=1 model_retrain=True model_retraining_epoch=80 patch_size=8 patching_strategy=random retraining_factor=4 selection_strategy=max-norm|0.1|0.053|0.64/0.6364|[result json](SleeperAgentAttack/MITM_2023-08-31T011535.882149/MITMPoisonSleeperAgent_1693444544.json)
