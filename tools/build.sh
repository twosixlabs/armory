#!/usr/bin/env bash
set -e

DRYRUN=
PUSH_IMAGES=
TAGGED_VERSION=
ARMORY_VERSION=`armory --show-docker-version-tag`


usage() {
    echo "usage: $0 [--dry-run|--push|--tag]" 1>&2
    exit 1
}


while [ "${1:-}" != "" ]; do
    case "$1" in
        -n|--dry-run) DRYRUN="echo" ;;
        -p|--push)    PUSH_IMAGES=1 ;;
        -t|--tag)     TAGGED_VERSION=$2; shift ;;
        *)            usage ;;
    esac
    shift
done


# TODO: Cache packages in a volume to speed up builds
# python -m pip download --destination-directory cache_dir -r pyproject.toml

# TODO: Install from cache_dir
# python -m pip install --no-index --find-links=cache_dir -r pyproject.toml

echo "Building the base image locally"
$DRYRUN docker build --force-rm --file Dockerfile -t twosixarmory/armory:${ARMORY_VERSION} --progress=auto .

echo "Retagging image as 'twosixarmory/armory:latest'"
$DRYRUN docker tag twosixarmory/armory:${ARMORY_VERSION} twosixarmory/armory:latest


if [[ -z "${TAGGED_VERSION}" ]]; then
    echo "Tagging image as 'twosixarmory/armory:${TAGGED_VERSION}'"
    $DRYRUN docker tag twosixarmory/armory:${ARMORY_VERSION} twosixarmory/armory:${TAGGED_VERSION}
fi


if [[ -z "${PUSH_IMAGES}" ]]; then
    echo "Pushing image to Docker Hub"
    $DRYRUN docker push twosixarmory/armory:${ARMORY_VERSION}
    $DRYRUN docker push twosixarmory/armory:latest
fi
