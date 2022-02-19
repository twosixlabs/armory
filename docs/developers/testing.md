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