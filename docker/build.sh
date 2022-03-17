#!/usr/bin/env bash

echo "docker/build.sh is no longer used and will be removed in armory version 0.16.0 (deprecated)"
echo "Please use the python script for building docker images. See:"
echo "    python docker/build.py --help"
exit 1

usage()
{
  cat <<EOF
armory docker build

This build script helps to build the docker images necessary for armory exectuion
(in docker mode).  The primary purpose of this script is to build the images from
scratch and to be used by the armory CI toolchain.

Armory uses scm versioning from git, therefore if you want to re-build containers
from a stable armory release, simply checkout that tag and then run this script

usage:
    docker/build.sh [options] framework

where framework is tf2, pytorch, or pytorch-deepspeech

OPTIONS
--base-tag=1.0.0      which twosixarmory/base tag to use (default: latest)
--no-cache            do not use local docker cache
-n | --dry-run        only show the build command that would be run
-h | --help           show this help
EOF
}

# Checking Script Execution Directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ "$PWD" != "$(dirname $SCRIPT_DIR)" ]; then
  echo "Must Execute build script from within armory/ folder that contains folder called \`docker\`"
  return 1
fi

# Setting Defaults
POSITIONAL_ARGS=()
NO_CACHE=false
ARMORY_VERSION=$(python setup.py --version | sed  -e 's/dev[0-9][0-9]*+//' -e 's/\.d[0-9][0-9]*$//')
DRYRUN=""
VERBOSE="--progress=auto"
REPO="twosixarmory"
FRAMEWORK=""
TAG=$ARMORY_VERSION
BASE_TAG=latest

# cannonicalize arguments: turn --opt=val to --opt val etc.
rewrite=$(getopt -o t:vnh --long tag:,base-tag:,verbose,dry-run,help,no-cache -n "$0" -- "$@")
[ $? -ne 0 ] && exit 1
# echo "rewrite $rewrite"
eval set -- "$rewrite"

while true ; do
    case "$1" in
        -t | --tag) TAG="$2"; shift 2;;
        --base-tag) BASE_TAG="$2"; shift 2;;
        -v | --verbose) verbose="--progress-plain" ; shift;;
        -n | --dry-run) dryrun=true ; shift ;;
        --no-cache) no_cache=true; shift ;;
        -h | --help) usage; shift ;;
        --) shift; break ;;
        *) echo "error unrecognized option $1" ; exit 1;;
    esac
done

FRAMEWORK="$1"
if [ -z "$FRAMEWORK" ]; then
  echo "$0: framework argument is required"
  usage
  exit 1
fi

case $FRAMEWORK in
    pytorch | pytorch-deepspeech | tf2) ;;
    *) echo "invalid framework: $FRAMEWORK should be one of pytorch, tf2, pytorch-deepspeech"; exit 1;;
esac

CMD="docker build"
if $NO_CACHE; then
    CMD="$CMD --no-cache"
fi

CMD="$CMD --force-rm"
CMD="$CMD --file $SCRIPT_DIR/Dockerfile-${FRAMEWORK}"
CMD="$CMD --build-arg base_image_tag=${BASE_TAG}"
CMD="$CMD --build-arg armory_version=${ARMORY_VERSION}"
CMD="$CMD --tag ${REPO}/${FRAMEWORK}:${TAG}"
CMD="$CMD $VERBOSE ."

if [ -n "$dryrun" ]; then
    echo "dry-run. would have executed: "
    echo "    ->  $CMD"
    exit 0;
fi

set +x
$CMD
