Contributing to Armory
======================
Contributions to Armory are welcomed and highly encouraged!  Armory contains a complex suite of tools that both configure the execution
environment and compose a set of objects (from an `experiment` file) to be executed in said environment.

Primarily, Armory has two main modes of operation:
  - Native (also known as `--no-docker`) mode - This uses a pre-set python environment to execute the configuration file.
  - Docker - This uses docker to compose and launch docker images, and executes the armory experiments
    within the container.

For more details, including how to set up your development environment for either mode of operation see: [Setting up Development Environment](#Setting-up-the-Development-Environment)

Armory Development follows the [GitHub Standard Fork & Pull Request Workflow](https://gist.github.com/Chaser324/ce0505fbed06b947d962).

Armory uses GitHub Actions to test contributions, for more details see [Armory CI](/.github/ci_test.yml).  Generally it will be most
useful to set up the [Armory pre-commit hooks](/tools/pre-commit.sh).  For more information see the [Armory Style Guide](/docs/style.md).

## Setting up the Development Environment
Armory follows the [GitHub Standard Fork & Pull Request Workflow](https://gist.github.com/Chaser324/ce0505fbed06b947d962) and therefore, to
get started with contributing to armory, you will first need to head over to [https://github.com/twosixlabs/armory](https://github.com/twosixlabs/armory)
and fork the repo.  Once forked, clone that fork to your computer and cd into the forked repo location (herein refered to as `YOUR_ARMORY_REPO`).

From here, you will need to setup your python virtual environment and, depending on your use case, other applications such as Docker.  The following
section will describe the details here in a bit more detail.

### Native Operation Mode
Armory can run natively within a python virtual environment on a `host` machine. To get setup you will need to
create a [virtual environment](https://docs.python.org/3/library/venv.html).  Once created and activated, you will need
to install some additional requirements.  Typically, it is useful to use the `-e` flag with the `armory` pip so that it
will point to your local directory, therefore utilizing code edits without requiring follow-on installs.  To accomplish
this run:
```bash
cd YOUR_ARMORY_REPO
pip install -e .[developer]
```
Now that you have the environment setup, kickoff the baseline tests to make sure its all good:
```bash
pytest -s tests/test_host
```
depending on you `$PATH`, pytest might refer to a pytest outside your virtualenv, which can cause issues.  As
an alternative you can use (make sure your virtualenv is active):
```bash
python -m pytest -s tests/test_host
```

If this is successful you are off to the races!  If you would like to run armory in `--no-docker` mode, see:
[Armory No Docker Setup](/docs/no_docker_mode.md).

### Docker Operation Mode
Armory can utilize [docker](https://www.docker.com/) to launch containers for execution of armory experiments.
For information on how to install docker on your machine see: [Docker Installation](https://docs.docker.com/get-docker/).

Once docker is installed, armory downloads and launches containers based on the `__version__` string found in `armory.__init__`.

Note: only release versions of armory will be published to [Dockerhub](https://hub.docker.com/), therefore,
development branch images much be built locally using:
```bash
cd YOUR_ARMORY_REPO
bash docker/build.sh <tf2|pytorch|pytorch-deepspeech|all> dev
```

## Style Guide
Armory enforces code / file styling using [Flake8](https://flake8.pycqa.org/), [black](https://github.com/psf/black),
[yamllint](https://yamllint.readthedocs.io/en/stable/), etc.  For more information about
how we configure these tools, see [Armory Style Guide](/docs/style.md).

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

## Test Cases
When adding new features please add test cases to ensure their correctness. We use
pytest as our test runner.

For running `pytest`, users should follow `.github/workflows/ci_test.yml`.
This has tests for docker and native modes as well as formatting.

## Documentation
When adding new functionality or modifying existing functionality, please update documentation.
Docs are all markdown (`.md`) files located in [docs](/docs/) directory or its subdirectories.
If doc files are added or removed, please also update the [markdown yaml](/mkdocs.yml)
