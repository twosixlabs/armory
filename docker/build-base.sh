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
        *)
            usage ;;
    esac
    shift
done

echo "Building the base image locally"
$dryrun docker build --force-rm --file ./docker/Dockerfile-base -t twosixarmory/base:latest --progress=auto .

if [[ -z "$push" ]]; then
    echo ""
    echo "If building the framework images locally, use the '--no-pull' argument. E.g.:"
    echo "    python docker/build.py all --no-pull"
    exit 0
fi

tag=$(python -m armory --version)
echo tagging twosixarmory/base:latest as $tag for dockerhub tracking
$dryrun docker tag twosixarmory/base:latest twosixarmory/base:$tag

echo ""
echo "If you have not run 'docker login', with the proper credentials, these pushes will fail"
echo "see docs/docker.md for instructions"
echo ""

# the second push should result in no new upload, it just tag the new image as
# latest
$dryrun docker push twosixarmory/base:$tag
$dryrun docker push twosixarmory/base:latest
