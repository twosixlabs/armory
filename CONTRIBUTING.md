# Development "-dev" Versions and Dockerfile

On the master branch and other non-release branches, the version number will end with `-dev`.
On master, this should start from the latest release version number.

These dev versions will not be published to PyPI or Dockerhub, and so must be build locally.

The primary docker images for testing and development should be built locally as follows:
```
version=$(python -c "import armory; print(armory.__version__)")
docker build --build-arg armory_version=${version} --target armory-tf1-dev -t twosixarmory/tf1:${version} .
docker build --build-arg armory_version=${version} --target armory-tf2-dev -t twosixarmory/tf2:${version} .
docker build --build-arg armory_version=${version} --target armory-pytorch-dev -t twosixarmory/pytorch:${version} .
```
NOTE: if the current version is not `-dev`, it may overwrite an existing release Dockerfile,
only using the local repository instead of pulling from PyPI. This is desired behavior,
and enables easier integration testing of release candidates before PyPI publication.

# Testing

Local testing with docker on a development branch should be done from the repo base directory using
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
