# Command Line Usage

## Root
* `armory <command> --root [...]`
Applies to `run`, `launch`, and `exec` commands.

This will run the docker container as root instead of the host user.
NOTE: this is incompatible with `--no-docker` mode.
NOTE: `--jupyter` only runs as root currently, and will ignore this argument.

### Example Usage

To run a single scenario as root:
```
armory run official_scenario_configs/cifar10_baseline.json --root
```

To accept a config file from standard in:
```
more official_scenario_configs/cifar10_baseline.json | armory run -
```

To execute the `id` command in the container:
```
$ python -m armory exec pytorch --root -- id
2020-07-15 15:02:20 aleph-5.local armory.docker.management[35987] INFO ARMORY Instance c1045b0ed3 created.
2020-07-15 15:02:20 aleph-5.local armory.eval.evaluator[35987] INFO Running bash command: id
uid=0(root) gid=0(root) groups=0(root)
...
```

## GPUs
* `armory <command> --gpus=X [...]`
* `armory <command> --use-gpu [...]`
Applies to `run`, `launch`, and `exec` commands.

This will specify whether to run GPUs and which ones to run.
If the `--gpus` flag is used, it will set `--use-gpu` to True.
The argument `X` for `--gpus` can be a single number, "all",
or a comma-separated list of numbers without spaces for multiple GPUs.

The `--use-gpu` flag will simply enable gpus.
If a config is being run, the gpus used will be pulled from the config.
If a config is not being run or that field is not in the config, it will default to all.

NOTE: when running a config, these will overwrite the fields inside the config.

### Example Usage

Examples:
```
armory run scenario_configs/mnist_baseline.json --use-gpu
armory launch tf2 --gpus=1,4 --interactive
armory exec pytorch --gpus=0 -- nvidia-smi
```

## Check Runs, Number of Example Batches, Indexing, and Class Filtering
* `armory run <config> --check [...]`
* `armory run <config> --num-eval-batches=X [...]`
* `armory run <config> --index=a,b,c [...]`
* `armory run <config> --classes=x,y,z [...]`
Applies to `run` command.

The `--check` flag will make every dataset return a single batch,
which is useful to quickly check whether the entire scenario correctly runs.
It will also ensure that the number of training epochs and certain attack parameters are set to 1.

The `--num-eval-batches` argument will truncate the number of batches used in
both benign and adversarial test sets.
It is primarily designed for attack development iteration, where it is typically unhelpful
to run more than 10-100 examples.

The `--index` argument will only use samples from the comma-separated, non-negative list of numbers provided.
Any duplicate numbers will be removed and the list will be sorted.
If indices beyond the size of the dataset are provided, an error will result at runtime.
Cannot be used with the `--num-eval-batches` argument.
Currently, batch size must be set to 1.

The `--classes` argument will only use samples from the comma-separated, non-negative list of numbers provided.
Any duplicate numbers will be removed and the list will be sorted.
If indices beyond the size of the dataset are provided, an error will result at runtime.
Can be used with `--index` argument. In that case, indexing will be done after class filtering.

NOTE: `--check` will take precedence over the `--num-eval-batches` argument.

### Example Usage

```
armory run scenario_configs/mnist_baseline.json --check
armory run scenario_configs/mnist_baseline.json --num-eval-batches=5
```

## Model Validation
The `--validate-config` flag will run a series of tests on the model in the selected configuration file.  These tests will alert the user to configuration errors (e.g. clip values that do not broadcast correctly to the input), as well as circumstances that may limit the evaluation (e.g. a model without gradients won't work with white box attacks without modification).

### Example Usage
```
armory run scenario_configs/so2sat_baseline.json --validate-config
```

## Skipping Benign Evaluation / Attack Generation
The `--skip-benign` and `--skip-attack` flags allow the user to skip, respectively, evaluating on benign samples and generating/evaluating attack samples.

### Example Usage
```
armory run scenario_configs/mnist_baseline.json --skip-benign
armory run scenario_configs/mnist_baseline.json --skip-attack
```

## Skipping Attack of Misclassified Samples
When `--skip-misclassified` is enabled, for benign examples that yield a misclassification, Armory will simply reuse the 
benign sample rather than running an attack. Note: the following criteria must be met when `--skip-misclassified` is enabled:

1. The scenario must be a classification task (i.e. *not* object detection, ASR) with the 'categorical_accuracy' metric enabled in the config file.
2. Batch size must be set to 1
3. The `--skip-benign` and `--skip-attack` flags cannot also be enabled

### Example Usage
```
armory run scenario_configs/mnist_baseline.json --skip-misclassified
```

## command line arguments and sysconfig

For convenience, command line control arguments can be specified in the "sysconfig"
block of an evaluation configuration. Adding control to the configuration is
described in [Configuration Files][conf]. Command line arguments will override
sysconfig specifications.


  [conf]: configuration_files.md#sysconfig-and-command-line-arguments


## CLI Utilities
* `armory utils <util-command>`
Additional armory utility functions are provided using the above command. See [utils.md](./utils.md) for more info.