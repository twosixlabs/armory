#!/usr/bin/env bash
# Build a dev image for specific framework. Optionally `all` frameworks can be built.
# Ex: `bash docker/build-dev.sh pytorch`

if [ "$#" -ne 1 ]; then
    echo "Please pass a single argument to specify which framework to build. Must be either \`tf1\`, \`tf2\` or \`pytorch\` or \`all\`"
    exit 1
fi

if [[ "$1" != "pytorch" && "$1" != "tf1" && "$1" != "tf2" && "$1" != "all" ]]; then
    echo "Framework argument must be either \`tf1\`, \`tf2\` or \`pytorch\` or \`all\`"
    exit 1
fi

# Parse Version
version=$(python -m armory --version)

if [[ "$1" == "all" ]]; then
  echo "Building docker images for all frameworks..."
  for framework in "tf1" "tf2" "pytorch"; do
    docker build --force-rm --file docker/Dockerfile --target armory -t twosixarmory/armory:${version} .
    docker build --force-rm --file docker/${framework}/Dockerfile --build-arg armory_version=${version} --target armory-${framework}-base -t twosixarmory/${framework}-base:${version} .
    docker build --force-rm --file docker/${framework}-dev/Dockerfile --build-arg armory_version=${version} --target armory-${framework}-dev -t twosixarmory/${framework}:${version} .
  done
else
  docker build --force-rm --file docker/Dockerfile --target armory -t twosixarmory/armory:${version} .
  docker build --force-rm --file docker/${1}/Dockerfile --build-arg armory_version=${version} --target armory-${1}-base -t twosixarmory/${1}-base:${version} .
  docker build --force-rm --file docker/${1}-dev/Dockerfile --build-arg armory_version=${version} --target armory-${1}-dev -t twosixarmory/${1}:${version} .
fi
