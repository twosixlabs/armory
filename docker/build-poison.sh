#!/usr/bin/env bash
# Build a docker image for specific framework. Optionally `all` frameworks can be built.
# Ex: `bash docker/build.sh pytorch`
if [ "$#" -ne 1 ]; then
    echo "Usage: bash docker/build-poison.sh <framework>"
    echo "    <framework> be \`tf1-poison\`, \`tf2-poison\`, \`pytorch-poison\`, or \`all\`"
    exit 1
fi

# Parse framework argument
if [[ "$1" != "pytorch-poison" && "$1" != "tf1-poison" && "$1" != "tf2-poison" && "$1" != "all" ]]; then
    echo "ERROR: <framework> argument must be \`tf1-poison\`, \`tf2-poison\`, \`pytorch-poison\`, or \`all\`, not \`$1\`"
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
for framework in "tf1-poison" "tf2-poison" "pytorch-poison"; do
    if [[ "$1" == "$framework" || "$1" == "all" ]]; then
        echo "\n\n\n"
        echo "------------------------------------------------"
        echo "Building docker image for framework: $framework"
        docker build --cache-from twosixarmory/armory:${version} --force-rm --file docker/${framework}/Dockerfile --build-arg armory_version=${version} -t twosixarmory/${framework}:${version} .
    fi
done
