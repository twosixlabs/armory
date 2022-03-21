echo "Building the base image locally"
docker build --force-rm --file ./docker/Dockerfile-base -t twosixarmory/base:latest --progress=auto .
echo "If building the framework images locally, use the '--no-pull' argument. E.g.:"
echo "    python docker/build.py all --no-pull"
