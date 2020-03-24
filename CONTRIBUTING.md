# Development "-dev" Versions and Dockerfile

On the master branch and other non-release branches, the version number will end with `-dev`.
On master, this should start from the latest release version number.

These dev versions will not be published to PyPI or Dockerhub, and so must be build locally.

The primary docker images for testing and development should be built locally as follows:
```
bash docker/build-dev.sh
```
NOTE: if the current version is not `-dev`, it may overwrite an existing release Dockerfile,
only using the local repository instead of pulling from PyPI. This is desired behavior,
and enables easier integration testing of release candidates before PyPI publication.

These should be rebuilt after any changes to the repo on the current branch.

# Testing

Local testing with docker on a development branch can also be done from the repo base directory using
```
python -m armory
```
instead of a pip-installed call
```
armory
```

This ensures that when the docker container is launched, the current branch is in the workspace,
which takes precedence over the pip installed version in the docker container.

A simple end-to-end integration test can be launched with
```
python -m armory run tests/test_data/fgm_attack_test.json
```

For running `pytest`, users should follow the `.github/workflows/ci_test.yml` process. This splits tests into those used on the host:
```
pytest -s --disable-warnings tests/test_host
```
and the rest, which are run in the container (after building):
```
version=$(python -m armory --version)
bash docker/build-dev.sh all
docker run -w /armory_dev twosixarmory/tf1:${version} bash \
    pytest -s --disable-warnings --ignore=tests/test_docker
```
