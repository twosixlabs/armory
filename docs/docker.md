# Docker
Armory is intended to be a lightweight python package which standardizes all evaluations
inside a docker container.


## Images
There are three docker images that are published to dockerhub for every release of 
the armory framework:

1. `twosixarmory/tf1:<version>` 
2. `twosixarmory/tf2:<version>` 
3. `twosixarmory/pytorch:<version>` 

When using `armory launch` or `armory exec` the framework specific arguments will 
utilize one of these three images. 

When running `armory run <path/to/config.json>` the image launched will be whatever is 
specified in the `docker_image` field. This enables users to extend our base images 
and run evaluations on an image that has all additional requirements for their defense.


### Custom Images

If you wish to utilize custom images for armory, these can be directly specified by
either the `"docker_image"` field of the [config file](docs/configuration_files.md)
of `armory run <path/to/config.json>` or in the CLI of the `launch` and `exec` commands,
as in `run launch <custom_image:tag>`.

Note: since Armory executes commands on detached containers, the `CMD` of the Docker image 
will be *ignored* and replaced with `tail -f /dev/null` to ensure that the container does not
exit while those commands are being executed.

### Interactive Use

As detailed [here](docs/index.md), it is possible to run the armory docker container in an
interactive mode using the `--interactive` CLI argument on `launch` or `run` commands.
We recommend this for debugging purposes, primarily.

When run, armory will output instructions for attaching to the container, similar to the following:
```
*** In a new terminal, run the following to attach to the container:
    docker exec -it -u 1001:1001 c10db6c70a bash
*** To gracefully shut down container, press: Ctrl-C
```
Note that `c10db6c70a` in this example is the container ID, which will change each time the
command is run. The `1001:1001` represents a mapping of users into the container, and will change
between systems and users. As stated, pressing `Ctrl-C` in that bash terminal will shut
down the container. To attach to the container, run the given command in a different bash terminal.

This will bring you into the docker container, and bring up a bash prompt there:
```
$ docker exec -it -u 1001:1001 c10db6c70a bash
groups: cannot find name for group ID 1001
I have no name!@c10db6c70a81:/workspace$
```
The groups error and the user name `I have no name!` may show up, depending on the host system, and
can be safely ignored. This is only due to host user not having a corresponding group ID inside
the container.

Once inside the container, you should be able to run or import armory as required:
```
I have no name!@c10db6c70a81:/workspace$ armory version
0.7.0-dev
I have no name!@c10db6c70a81:/workspace$ python
Python 3.7.6 (default, Jan  8 2020, 19:59:22) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import armory
>>> 
```

Note: We do not recommend using `--interactive` mode for installing custom requirements. You may 
run into permissions issues, as everything is installed as root, but the armory user is not run
as root, to prevent potential security issues. Instead, we recommend creating a custom Docker image,
as described above.

## Building Images from Source
When using a released version of armory, docker images will be pulled as needed when 
evaluations are ran. However if there are issues downloading the images (e.g. proxy) 
they can be built from the release branch of the repo:
```
git checkout -b r0.5
bash docker/build.sh <tf1|tf2|pytorch|all>
```

## Docker Volume Mounts
When launching an ARMORY instance several host directories will be mounted within the 
docker container. Note, the host directory path for datasets, saved_models, and 
outputs are configurable. To modify those directories simply run `armory configure`.
The defaults are shown below:


| Host Path   | Docker Path   | 
|:----------: | :-----------: | 
| os.getcwd() | /workspace    |  
| ~/.armory/datasets | /armory/datasets |
| ~/.armory/saved_models | /armory/saved_models |
| ~/.armory/outputs | /armory/outputs |

<br>

When using these paths in code, armory provides a programatic way to access these 
directories.

#### Utilizing the paths
Code that runs on host (e.g debugging outside of container):
```
from armory import paths
paths.host().dataset_dir
```

Code that runs in container (e.g within an evaluation):
```
from armory import paths
paths.docker().dataset_dir
```

## Using GPUs with Docker
Armory uses the nvidia runtime to use GPUs inside of Docker containers.

This can be specified in JSON config files with "sysconfig" as follows:
```
    ...
    "sysconfig": {
        ...
        "gpus": "7",
        "use_gpu": true
    }
    ...
```
The `use_gpu` flag takes a boolean true/false value, and specifies whether to use the gpu or default to cpu.
The `gpus` flag is optional, and is ignored if `use_gpu` is false. If `use_gpu` is true, it defaults to using all GPUs.
    If present, the value should be a `,`-separated list of numbers specifying the GPU index in `nvidia-smi`.
    For instance, `"gpus": "2,4,7"` would enable three GPUs with indexes 2, 4, and 7.
    Setting the field to be `all` will enable use of all available gpus, i.e. `"gpus": "all"` will enable all GPUs.

## Docker Setup
Depending on the evaluation, you may need to increase the default memory allocation for 
docker containers on your system. 

Linux does not limit memory allocation, but on Mac and Windows this defaults to 2 GB 
which is likely insufficient. See the docker documentation to change this:
* [Mac](https://docs.docker.com/docker-for-mac/)
* [Windows](https://docs.docker.com/docker-for-windows/)


## Docker Image Maintenance 
Since there are new docker images for every release of ARMORY, you may want to clean up
your docker image cache as you increase versions.

To download the current version's images ,and remove old ones, simply run:
```
armory clean --force
```

To display the set of current images on your machine, you can run:
```
docker images
```
To manually delete images, see the docs for [docker rmi](https://docs.docker.com/engine/reference/commandline/rmi/).


### Docker Container Maintenance
In order to see the set of containers that are running, run:
```
docker ps
```
ARMORY will attempt to gracefully shut down all containers it launches;
however, certain errors may prevent shutdown and leave running containers.
To shut down these containers, please see the docs for 
[docker stop](https://docs.docker.com/engine/reference/commandline/stop/) 
and [docker kill](https://docs.docker.com/engine/reference/commandline/kill/).

## Running without docker

Armory has partial support for users wishing to run without docker. Currently, the
`armory run` command can be run without Docker in Linux environments. To run without
docker, either set the `docker_image` field to be null in the scenario
configuration json file, or call `armory run` with the --no-docker option.

### Environment setup
The listing of libraries needed for Armory when run on host is available at
`host-requirements.txt`. 