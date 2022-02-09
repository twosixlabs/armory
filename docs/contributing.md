Contributing to Armory
======================
Contributions to Armory are welcomed and highly encouraged!  Armory contains a complex suite of tools that both configure the execution
environment as well as compose a set of objects (from a configuration file) to be executed in said environment.

Primarily, Armory has two main modes of operation:
  - Native (also known as `--no-docker`) mode - This uses a pre-set python environment to execute the configuration file
  - Docker - In this mode of operation, armory composes a Docker image, launches the image, and executed the armory configuration
    within the image.
    
For more details, including how to setup your development environment for either mode of operation see: [Setting up Development Environment](#Setting-up-the-Development-Environment)

Armory Development follows the [GitHub Standard Fork & Pull Request Workflow](https://gist.github.com/Chaser324/ce0505fbed06b947d962).  

Armory uses GitHub Actions to test contributions, for more details see [Armory CI](../.github/ci_test.yml).  Generally it will be most 
useful to setup the [Armory pre-commit hooks](../tools/pre-commit.sh).  For more information see the [Armory Style Guide](../STYLE.md)

## Setting up the Development Environment
```bash
pip install -r test-requirements.txt
```
### Docker Operation Mode

### Native Operation Mode
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
## Style Guide
Armory enforces code / file styling using [Flake8](https://flake8.pycqa.org/), [black](https://github.com/psf/black),
[yamllint](https://yamllint.readthedocs.io/en/stable/), etc.  For more information about
how we configure these tools, see [Armory Style Guide](../STYLE.md).


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
