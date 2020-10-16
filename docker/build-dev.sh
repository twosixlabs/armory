#!/usr/bin/env bash
# Build a dev docker image for specific framework. Optionally `all` frameworks can be built.
# Ex: `bash docker/build-dev.sh pytorch`

if [ "$#" -ne 1 ]; then
    echo "Usage: bash docker/build-dev.sh <framework>"
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

# Build images
echo "Building base docker image: armory"
docker build --force-rm --file docker/Dockerfile --target armory -t twosixarmory/armory:${version} .
for framework in "tf1" "tf2" "pytorch"; do
    if [[ "$1" == "$framework" || "$1" == "all" ]]; then
        echo "Building docker images for framework: $framework"
        docker build --force-rm --file docker/${framework}/Dockerfile --build-arg armory_version=${version} --target armory-${framework}-base -t twosixarmory/${framework}-base:${version} .
        docker build --force-rm --file docker/${framework}-dev/Dockerfile --build-arg armory_version=${version} --target armory-${framework}-dev -t twosixarmory/${framework}:${version} .
    fi
done

if [[ "$1" == "pytorch" || "$1" == "all" ]]; then
    echo "Building docker images for deep speech model"
    docker build --force-rm --file docker/pytorch-deepspeech-dev/Dockerfile --build-arg armory_version=${version} --target armory-pytorch-deepspeech-dev -t twosixarmory/pytorch-deepspeech:${version} .
fi
