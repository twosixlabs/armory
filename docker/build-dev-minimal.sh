#!/usr/bin/env bash
# Build a single dev image for a specific framework. Single argument should be passed.
# Ex: `bash docker/build-dev-minimal.sh pytorch`

if [ "$#" -ne 1 ]; then
    echo "Please specify a single argument to specify which framework to build. Must be either \`tf1\`, \`tf2\` or \`pytorch\`"
    exit 1
fi

if [[ "$1" != "pytorch" && "$1" != "tf1" && "$1" != "tf2" ]]; then
    echo "Framework argument must be either \`tf1\`, \`tf2\` or \`pytorch\`"
    exit 1
fi

version=$(python -m armory --version)
docker build --file docker/Dockerfile --target armory -t twosixarmory/armory:${version} .
docker build --file docker/${1}/Dockerfile --build-arg armory_version=${version} --target armory-${1}-base -t twosixarmory/${1}-base:${version} .
docker build --file docker/${1}-dev/Dockerfile --build-arg armory_version=${version} --target armory-${1}-dev -t twosixarmory/${1}:${version} .