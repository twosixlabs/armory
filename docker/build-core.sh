#!/usr/bin/env bash
set -e

usage() { echo "usage: $0 [--dry-run]" 1>&2; exit 1; }

dryrun=

while [ "${1:-}" != "" ]; do
    case "$1" in
        -n|--dry-run)
            echo "dry-run requested. not building or tagging"
            dryrun="echo" ;;
        *)
            usage ;;
    esac
    shift
done

echo "Building the core image locally"
$dryrun docker build --force-rm --file ./docker/Dockerfile -t twosixarmory/core:latest --progress=auto .

tag=$(python -m armory --version)
echo Tagging twosixarmory/core:latest as $tag for dockerhub tracking
$dryrun docker tag twosixarmory/core:latest twosixarmory/core:$tag
