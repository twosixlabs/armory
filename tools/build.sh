#!/usr/bin/env bash
set -e

DRYRUN=
PUSH_IMAGES=
ARMORY_VERSION=`armory --show-docker-version-tag`


usage() {
    echo "usage: $0 [--dry-run|--push]" 1>&2
    exit 1
}


while [ "${1:-}" != "" ]; do
    case "$1" in
        -n|--dry-run) DRYRUN="echo" ;;
        -p|--push)    PUSH_IMAGES=1 ;;
        *)            usage ;;
    esac
    shift
done

# TODO: Cache packages in a volume to speed up builds
# python -m pip download --destination-directory cache_dir -r pyproject.toml

# TODO: Install from cache_dir
# python -m pip install --no-index --find-links=cache_dir -r pyproject.toml

echo "Building the base image locally"
$DRYRUN docker build --force-rm --file Dockerfile -t twosixarmory/armory:${$ARMORY_VERSION} --progress=auto .


if [[ -z "${PUSH_IMAGES}" ]]; then
    echo "Retagging image as 'twosixarmory/armory:latest'"
    docker tag twosixarmory/armory:${$ARMORY_VERSION} twosixarmory/armory:latest
    echo "Pushing image to Docker Hub"
    docker push twosixarmory/armory:${$ARMORY_VERSION}
    docker push twosixarmory/armory:latest
fi
