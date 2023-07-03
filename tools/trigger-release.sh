#!/bin/bash

# conveience script to trigger a build and release of Armory
# requires the environment variable ARMORY_GITHUB_TOKEN to be set


if [ -z "$ARMORY_GITHUB_TOKEN" ] ; then
    echo "Environment variable ARMORY_GITHUB_TOKEN is not set"
    exit 1
fi


read -e -p  "Are you sure you want to trigger a Armory build and release (y/n) " choice
if [[ "$choice" != [Yy]* ]] ; then
    echo "Operation Aborted."
    exit 2
fi

echo "Sending request to trigger build and release..."

curl \
    -H "Accept: application/vnd.github.everest-preview+json" \
    -H "Authorization: token $ARMORY_GITHUB_TOKEN" \
    --request POST \
    --data '{"event_type": "build-and-release"}' \
    https://api.github.com/repos/twosixlabs/armory/dispatches

echo "Request sent. Check PYPI and Docker Hub for new releases."
