#!/usr/bin/env bash
# Build a docker image for specific framework. Optionally `all` frameworks can be built.
# Ex: `bash docker/build.sh pytorch`
if [ "$#" -ne 1 ]; then
    echo "Usage: bash docker/build.sh <framework>"
    echo "    <framework> be \`armory\`, \`tf1\`, \`tf2\`, \`pytorch\`, or \`all\`"
    exit 1
fi

# Parse framework argument
if [[ "$1" != "armory" && "$1" != "pytorch" && "$1" != "tf1" && "$1" != "tf2" && "$1" != "all" ]]; then
    echo "ERROR: <framework> argument must be \`armory\`, \`tf1\`, \`tf2\`, \`pytorch\`, or \`all\`, not \`$1\`"
    exit 1
fi

# Parse Version
version=$(python -m armory --version)
if [[ $version == *"-dev" ]]; then
    echo "Armory version $version is a '-dev' branch. To build docker images, use:"
    echo "bash docker/build-dev.sh"
    exit
fi

# Build images
echo "\n\n\n"
echo "------------------------------------------------"
echo "Building base docker image: armory"
docker build --cache-from twosixarmory/armory:latest --force-rm --file docker/Dockerfile --target armory -t twosixarmory/armory:${version} .
for framework in "tf1" "tf2" "pytorch"; do
    if [[ "$1" == "$framework" || "$1" == "all" ]]; then
        echo "\n\n\n"
        echo "------------------------------------------------"
        echo "Building docker image for framework: $framework"
        docker build --cache-from twosixarmory/${framework}:latest,twosixarmory/armory:${version} --force-rm --file docker/${framework}/Dockerfile --build-arg armory_version=${version} --target armory-${framework} -t twosixarmory/${framework}:${version} .
    fi
done

if [[ "$1" == "pytorch" || "$1" == "all" ]]; then
    echo "Building docker image for deep speech model"
    docker build --cache-from twosixarmory/armory:latest --force-rm --file docker/pytorch/Dockerfile --build-arg armory_version=${version} --target armory-pytorch-base -t twosixarmory/pytorch-base:${version} .
    docker build --cache-from twosixarmory/pytorch-deepspeech:latest,twosixarmory/armory:${version} --force-rm --file docker/pytorch-deepspeech/Dockerfile --build-arg armory_version=${version} --target armory-pytorch-deepspeech -t twosixarmory/pytorch-deepspeech:${version} .
fi
