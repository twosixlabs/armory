
# Docker
Armory is intended to be a lightweight python package which standardizes all evaluations
inside a docker container. Docker images will be pulled as needed when evaluations are 
ran.

However if there are issues downloading the images (e.g. proxy) they can be built 
from the repo:
```
bash docker/build.sh <tf1|tf2|pytorch|all>
```

NOTE: if you're a developer working on armory from the master branch, you will need to 
build the dev containers: 
```
bash docker/build-dev.sh <tf1|tf2|pytorch|all>
```

### Docker Mounts
By default when launching an ARMORY instance the current working directory will be mounted
as your default directory.This enables users to run modules from ARMORY baselines, 
as well as modules from the user project.

### Docker Setup
Depending on the task, docker memory for an ARMORY container must be at least 8 GB to run properly (preferably 16+ GB).
On Mac and Windows Desktop versions, this defaults to 2 GB. See the docs to change this:
* [Mac](https://docs.docker.com/docker-for-mac/)
* [Windows](https://docs.docker.com/docker-for-windows/)

### Docker Cleanup
Running `armory download-all-data` will download new Docker images, but will not clean up old images.

To download new images and clean up old images:
```
armory clean
```
If containers are currently running that use the old images, this will fail.
In that case, either stop them with first or run:
```
armory clean --force
```

To display the set of current images:
```
docker images
```
To manually delete images, see the docs for [docker rmi](https://docs.docker.com/engine/reference/commandline/rmi/).

In order to see the set of containers that are running:
```
docker ps
```
ARMORY will attempt to gracefully shut down all containers it launches;
however, certain errors may prevent shutdown and leave running containers.
To shut down these containers, please see the docs for [docker stop](https://docs.docker.com/engine/reference/commandline/stop/) and [docker kill](https://docs.docker.com/engine/reference/commandline/kill/).
