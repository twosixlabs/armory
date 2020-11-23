#!/usr/bin/env bash
# Build a dev docker image for specific framework. Optionally `all` frameworks can be built.
# Base images for the specified framework must already exist.
# Ex: `bash docker/build-poison-dev.sh pytorch-poison` after creating base pytorch image.

if [ "$#" -ne 1 ]; then
    echo "Usage: bash docker/build-poison-dev.sh <framework>"
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
for framework in "tf1-poison" "tf2-poison" "pytorch-poison"; do
    if [[ "$1" == "$framework" || "$1" == "all" ]]; then
        echo "Building docker images for framework: $framework"
        docker build --force-rm --file docker/${framework}-dev/Dockerfile --build-arg armory_version=${version} -t twosixarmory/${framework}:${version} .
    fi
done
