#!/usr/bin/env bash
set -e

usage() { echo "usage: $0 [--dry-run] [--push]" 1>&2; exit 1; }

dryrun=
push=

while [ "${1:-}" != "" ]; do
    case "$1" in
        -n|--dry-run)
            echo "dry-run requested. not building or pushing to docker hub"
            dryrun="echo" ;;
        --push)
            push=true ;;
        --cached)
            cached=true ;;
        *)
            usage ;;
    esac
    shift
done

if [[ -z "${cached}" ]]; then
    ARMORY_DOCKERFILE="./docker/Dockerfile-base"

    $dryrun docker build \
        --file "${ARMORY_DOCKERFILE}" \
        --target armory-base \
        -t twosixarmory/armory-base:latest \
        --progress=auto \
        .

    $dryrun docker build \
        --file "${ARMORY_DOCKERFILE}" \
        --target armory-build \
        --cache-from=twosixarmory/armory-base:latest \
        -t twosixarmory/base-build:latest \
        --progress=auto \
        .

    $dryrun docker build \
        --file "${ARMORY_DOCKERFILE}" \
        --target armory-release \
        --cache-from=twosixarmory/armory-build:latest \
        -t twosixarmory/base:latest \
        --progress=auto \
        .

    # docker scan --accept-license --dependency-tree --file "${ARMORY_DOCKERFILE}" twosixarmory/base:latest
    # docker system prune --all --force
    # docker run --gpus all --rm -it -v `pwd`:/tmp --entrypoint /bin/bash twosixarmory/base:latest

else
    echo "Building the base image locally"
    $dryrun docker build --force-rm --file ./docker/Dockerfile-base -t twosixarmory/base:latest --progress=auto .
fi


# if [[ -z "$push" ]]; then
#     echo ""
#     echo "If building the framework images locally, use the '--no-pull' argument. E.g.:"
#     echo "    python docker/build.py all --no-pull"
#     exit 0
# fi

# tag=$(python -m armory --version)
# echo tagging twosixarmory/base:latest as $tag for dockerhub tracking
# $dryrun docker tag twosixarmory/base:latest twosixarmory/base:$tag

# echo ""
# echo "If you have not run 'docker login', with the proper credentials, these pushes will fail"
# echo "see docs/docker.md for instructions"
# echo ""

# # the second push should result in no new upload, it just tag the new image as
# # latest
# $dryrun docker push twosixarmory/base:$tag
# $dryrun docker push twosixarmory/base:latest
