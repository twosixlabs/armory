Running Armory Tests
=========================

Tests have to download a bunch of code (external repos,etc.) and model weights the first
time around so that one can take a while.  

You will need to have the `ARMORY_GITHUB_TOKEN` env variable set (which may be done by 
armory configure...but will need to make sure)

Can use pytest -s to run all tests:
```bash
pytest -s ./tests/`
```

To only run a single file:
```bash
pytest -s ./tests/test_file.py
```
or to run only a single tests
```bash
pytest -s ./tests/test_file.py::test_name
```

If a test is parameterized to see how to only run one of the 
parameters sets use:
```bash
pytest --collect-only -q 
```
Then run the one you want (for example):
```bash
 pytest -s tests/test_models.py::test_model_creation[armory.baseline_models.pytorch.cifar-get_art_model-None-cifar10-500-1-1-100-1-numpy-0.25]
```

## Running pytest in Docker

When running pytest with docker, you have two choices.

First, you can rebuild the docker container and then run pytest with the container:
```bash
python docker/build.py -f pytorch --no-pull
armory exec pytorch -- pytest -s ./tests/
```
The `armory exec pytorch` is equivalent to launching an interactive container with `armory launch pytorch`, bashing into the container, and running `pytest -s ./tests/`.

Or, if rebuilding the container is onerous, you can just do the `armory exec` command, but you need to make sure that pytest is invoked with `python -m`.
Otherwise, tests will import armory installed in the container, not your locally modified dev version.
```bash
armory exec pytorch -- python -m pytest -s ./tests/
```
