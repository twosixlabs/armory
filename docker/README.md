# Docker
Armory is intended to be a lightweight python package which standardizes all evaluations
inside a docker container.

## Updates
  - As of Armory version 0.16.0 the `docker/build.sh` script has been deprecated.
  - Standardized python build script: `python docker/build.py --help`
  - As of Armory version 0.15.0, we no longer support or publish a `tf1` image.
  - If `tf1` functionality is needed, please use the `tf2` image and use `tf1` compatibility mode.


### Custom Images
If you wish to utilize custom images for armory, these can be directly specified by
either the `"docker_image"` field of the [config file](configuration_files.md)
of `armory run <path/to/config.json>` or in the CLI of the `launch` and `exec` commands,
as in `run launch <custom_image:tag>`.

Note: since Armory executes commands on detached containers, the `CMD` of the Docker image
will be *ignored* and replaced with `tail -f /dev/null` to ensure that the container does not
exit while those commands are being executed.


## Building Images from Source
When using a released version of armory, docker images will be pulled as needed when
evaluations are ran. However if there are issues downloading the images (e.g. proxy)
they can be built from the release branch of the repo:
```
git checkout -b r0.16.0
bash docker/build-base.sh
python docker/build.py <armory|pytorch-deepspeech|all> [--no-pull]
```

If possible, we recommend downloading the base image instead of building, which can be done by removing the `--no-pull` argument from `build.py`.


## Docker Volume Mounts
Host directory path for datasets, saved_models, and outputs are configurable. To modify those directories simply run `armory configure`.
The defaults are shown below:


| Host Path   | Docker Path   |
|:----------: | :-----------: |
| os.getcwd() | /workspace    |
| ~/.armory/datasets | /armory/datasets |
| ~/.armory/saved_models | /armory/saved_models |
| ~/.armory/outputs | /armory/outputs |


When using these paths in code, armory provides a programatic way to access these
directories.


## Using GPUs with Docker
Armory uses the nvidia runtime to use GPUs inside of Docker containers.

### CUDA
Armory docker images currently use CUDA 11.6 as the base image ( see [Dockerfile-Base](../docker/Dockerfile-base))
and the TensorFlow versions we support require CUDA 10+. Previous versions of CUDA (e.g. CUDA<11.6) are not actively tested
by armory developers or CI tools.  However, if previous versions of CUDA are needed, the following instructions should
provide a decent starting point.

To use CUDA 10.2, you will need to rebuild the base image and the derived images with the following changes:
in [docker/Dockerfile-base](../docker/Dockerfile-base) change:
```bash
FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04
```
to
```bash
FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
```
and then change `cudatoolkit=11.6 \` to `cudatoolkit=10.2 \`.

Again, this is not actively tested, so it may require further modification of library dependencies to
work appropriately. Also, while PyTorch does support CUDA 9, we do not provide support in armory due to
TFDS dependencies and we do not recommend using versions less than the standard 11.6.


## Docker Setup
Depending on the evaluation, you may need to increase the default memory allocation for
docker containers on your system.

Linux does not limit memory allocation, but on Mac and Windows this defaults to 2 GB
which is likely insufficient. See the docker documentation to change this:
* [Mac](https://docs.docker.com/docker-for-mac/)
* [Windows](https://docs.docker.com/docker-for-windows/)

### Environment setup
NOTE: The listing of libraries needed for Armory when run on host is available at
`pyproject.toml`. You will need to manually install the requirements in
that file that match your framework (TF1, TF2, PyTorch).
