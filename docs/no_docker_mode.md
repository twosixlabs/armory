Armory No-Docker Mode 
=======================
In order to run armory in `--no-docker` mode, you will need a properly
setup environment.  Generally folks have used conda in the past, however this
document only requires a python (>=3.7, <3.9) environment to get going.

First you will need to clone the armory repo (or if you plan to be a developer, 
see [Contributing to Armory](./contributing.md)):
```bash 
git clone https://github.com/twosixlabs/armory.git
cd armory
```
Now you will want to create a clean python virtual environment and activate
that environment.  For example:
```bash
python37 -m venv venv37
source venv37/bin/activate
```
Once this is complete, and you have ensured you are in the `armory` directory, 
you can setup the environment with the following:
```bash
pip install --upgrade pip==22.0.3
pip install -r no-docker-req.txt
pip install -e . 
```
With this complete, you now can run armory using `armory run -h`.  If you would 
like to test the installation / environment, we have provided some base tests that
can be executed using:
```bash
pytest -s ./tests/unit/test_no_docker.py
```

This runs a series of configs in a variety of ways to ensure that 
the environment is operating as expected.  

## Run baseline CIFAR-10 config
Now to see if everything is operating correctly you can run a config.
For example the [CIFAR-10 Short](../scenario_configs/no_docker/cifar_short.json).
can be run with:
```bash
armory run ./scenario_configs/no_docker/cifar_short.json --no-docker
```
