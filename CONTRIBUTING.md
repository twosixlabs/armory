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
