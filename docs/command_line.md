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
armory launch tf1 --gpus=1,4 --interactive
armory exec pytorch --gpus=0 -- nvidia-smi
```

## Check Runs and Number of Examples
* `armory run <config> --check [...]`
* `armory run <config> --num-eval-batches=X [...]`
Applies to `run` command.

The `--check` flag will make every dataset return a single batch,
which is useful to quickly check whether the entire scenario correctly runs.
It will also ensure that the number of training epochs is set to 1.

The `--num-eval-batches` argument will truncate the number of batches used in
both benign and adversarial test sets.
It is primarily designed for attack development iteration, where it is typically unhelpful
to run more than 10-100 examples.

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
