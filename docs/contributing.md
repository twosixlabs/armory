# Contributing to Armory
Contributions to Armory are welcomed and highly encouraged!  Generally, developing armory requires a different setup and some additional tools
as compared to just using armory to run configurations.  

## Setting up the Development Environment
To get setup you will need to create a [virtual environment](https://docs.python.org/3/library/venv.html).  Once created and activated, you will need 
to install the testing requirements.  Additionally its often useful to use the `-e` flag so that pip points to your local directory for the code
```bash
pip install -e .[tests]
pip install -r host-requirements.txt
```
Now that you have the environment setup, kickoff the baseline tests to make sure its all good:
```bash
pytest -s tests/test_host
```
If this is successful you are off to the races!

## Pull Requests

We gladly welcome [pull requests](
https://help.github.com/articles/about-pull-requests/).

If you've never done a pull request before we recommend you read
[this guide](http://blog.davidecoppola.com/2016/11/howto-contribute-to-open-source-project-on-github/) 
to get you started.

Before making any changes, we recommend opening an issue (if it
doesn't already exist) and discussing your proposed changes. This will
let us give you advice on the proposed changes. If the changes are
minor, then feel free to make them without discussion.


## Development Docker Containers
Armory launches containers based on the `__version__` string found in `armory.__init__`.

Only release versions will be published to Dockerhub, so development branch images much be built locally:
```
bash docker/build.sh <tf1|tf2|pytorch|pytorch-deepspeech|all> dev
```

## Test Cases
When adding new features please add test cases to ensure their correctness. We use 
pytest as our test runner. 

For running `pytest`, users should follow the `.github/workflows/ci_test.yml` process. 
This splits tests into those used on the host:
```
pytest -s --disable-warnings tests/test_host
```
and the rest, which are run in the container (after building):
```
python -m armory exec tf1 -- pytest -s tests/test_tf1/
python -m armory exec tf2 -- pytest -s tests/test_tf2/
python -m armory exec pytorch -- pytest -s tests/test_pytorch/
```

#### Example
As an example if you added a new pytorch baseline model, 
you'll want to add testcases to `tests/test_pytorch/test_pytorch_models.py`

To run simply use the armory exec functionality:
```
python -m armory exec pytorch -- pytest -s tests/test_pytorch/
```
