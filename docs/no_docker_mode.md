Armory No-Docker Mode
=======================
In order to run armory in `--no-docker` mode, you will need a properly
setup environment.  Generally folks have used conda in the past, however this
document only requires a python (>=3.7, <3.9) environment to get going.

First you will need to clone the armory repo (or if you plan to be a developer,
see [Contributing to Armory](./contributing.md) to clone a fork of the repo).
For the following, the repo directory will be referred to as `[armory-repo]`.

Now you will want to create a clean python virtual environment and activate
that environment.  For example:
```bash
cd [armory-repo]
python37 -m venv venv37
source venv37/bin/activate
```
or you can setup your virtualenv using another tool (e.g. conda, pyenv, etc.).

Once this is complete, and you have ensured you are in the `[armory-repo]` directory,
you can setup the environment with the following:
```bash
pip install --upgrade pip==22.0.3
pip install -e .[engine,datasets,math,pytorch,deepspeech,tensorflow]
```
Once this completes, you should run `armory configure` (If you haven't already done this
previously) to setup the armory configuration
(e.g. dataset download directory, output directory, etc.).

With this complete, you now can run armory using `armory run -h`.  If you would
like to test the installation / environment, we have provided some base tests that
can be executed using:
```bash
pytest -s ./tests/unit/test_no_docker.py
```

This runs a series of configs in a variety of ways to ensure that
the environment is operating as expected.

NOTE: If you run into issues running pytest (e.g. sometimes your `$PATH` is configured
to point to a global pytest that is outside your virtualenv) directly, you can use the
alternative approach (make sure your virtualenv is active):
```bash
python -m pytest -s ./tests/unit/test_no_docker.py
```

If you would like to run the example interactively you
enter a python session in the virtualenv and type:
```python
from armory.scenarios.main import get as get_scenario
from armory import paths
from pathlib import Path

# Armory needs to have the paths set correctly
paths.set_mode("host")

config = Path("scenario_configs/no_docker/cifar_short.json")
s = get_scenario(config).load()
s.evaluate()
```

## Run baseline CIFAR-10 config

Now to see if everything is operating correctly you can run the config file
of your choice.  The two provided below are truncated in their execution to
demonstrate functionality of armory and, therefore, will not produce accurate
results.  For more accurate results (and potentially longer running times) please
see [Armory Baseline Scenario Configs](../scenario_configs/)

#### [CIFAR-10 Short](../scenario_configs/no_docker/cifar_short.json).

```bash
armory run ./scenario_configs/no_docker/cifar_short.json --no-docker
```

#### [CARLA Short](../scenario_configs/no_docker/carla_short.json).

```bash
armory run ./scenario_configs/no_docker/carla_short.json --no-docker
```
