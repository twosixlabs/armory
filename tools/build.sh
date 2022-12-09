#!/usr/bin/env bash
set -e

usage() { echo "usage: $0 [--dry-run]" 1>&2; exit 1; }

dryrun=

while [ "${1:-}" != "" ]; do
    case "$1" in
        -n|--dry-run)
            echo "dry-run requested. not building or pushing to docker hub"
            dryrun="echo" ;;
        *)
            usage ;;
    esac
    shift
done

echo "Building the base image locally"
$dryrun docker build --force-rm --file Dockerfile -t twosixarmory/armory:latest --progress=auto .

if [[ -z "$push" ]]; then
    echo ""
    echo "If building the framework images locally, use the '--no-pull' argument. E.g.:"
    echo "    python docker/build.py all --no-pull"
    exit 0
fi
