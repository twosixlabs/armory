#!/usr/bin/env bash
# Build a docker image for specific framework. Optionally `all` frameworks can be built.
# Ex: `bash docker/build.sh pytorch`

Help()
{
  # Display Help
  echo "----------------------------------------------------------------------------------------"
  echo "                   Armory Docker Build Script                       "
  echo
  echo "This build script helps to build the necessary docker images to "
  echo "execute the typical armory experiments.  "
  echo
  echo "Syntax: build.sh [-f|t|nc|av|h]"
  echo "options:"
  echo "-f | --framework            Select Which framework to target:"
  echo "                                [all|base|tf2|pytorch|pytorch-deepspeech]"
  echo "-t | --tag                  Tag to use when creating Docker Image (e.g. \`latest\` in"
  echo "                                repo/image:latest"
  echo "-bt | --base-tag            Tag to use for Base Image"
  echo "-nc | --no-cache            Build Images \"Clean\" (i.e. using --no-cache)"
  echo "-av | --armory-version      Select armory version to install in container"
  echo "                              (default: local)"
  echo "-dr | --dry-run             Only show the Build calls (do not execute the builds)"
  echo "-v | --verbose              Show Logs in Plain Text (uses \`--progress=plain\`"
  echo "-r | --repo                 Set the Docker Repository"
  echo "-h | --help                 Print this Help."
  echo
  echo "----------------------------------------------------------------------------------------"
}


POSITIONAL_ARGS=()
CLEAN=false
NO_CACHE=false
TAG="dev"
ARMORY_VERSION="local"
DRYRUN=false
VERBOSE="--progress=auto"
REPO="twosixarmory"
BASE_TAG="dev"
FRAMEWORK=""
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ "$PWD" != "$(dirname $SCRIPT_DIR)" ]; then
  echo "Must Execute build script from within armory/ folder that contains Dockerfiles"
  return 1
fi

while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--framework)
      FRAMEWORK="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--tag)
      TAG="$2"
      shift # past argument
      shift # past value
      ;;
    -nc|--no-cache)
      NO_CACHE=true
      shift # past argument
      ;;
    -av|--armory-version)
      ARMORY_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    -dr|--dry-run)
      DRYRUN=true
      shift # past argument
      ;;
    -v|--verbose)
      VERBOSE="--progress=plain"
      shift # past argument
      ;;
    -r|--repo)
      REPO="$2"
      shift # past argument
      shift # past value
      ;;
    -bt|--base-tag)
      BASE_TAG="$2"
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      Help
      return
      ;;
    -*|--*)
      echo "Unknown option $1"
      echo "For more info try: build.sh -h"
      return 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters


if [ "$FRAMEWORK" == "" ]; then
  echo "Must Specify Framework"
  return 1
fi

if [[ "$FRAMEWORK" != "pytorch" &&  "$FRAMEWORK" != "pytorch-deepspeech" && "$FRAMEWORK" != "tf2" && "$FRAMEWORK" != "all"  && "$FRAMEWORK" != "base" ]]; then
    echo "ERROR: <framework> argument must be \`tf2\`, \`pytorch\`, \`pytorch-deepspeech\`, \`base\` or \`all\`, not \`$FRAMEWORK\`"
    return 1
fi

if [ $FRAMEWORK == "all" ]; then
  echo "setting all frameowks"
  FRAMEWORK=("base" "pytorch" "tf2" "pytorch-deepspeech")
else
  FRAMEWORK=($FRAMEWORK)
fi

if [ $ARMORY_VERSION == "local" ]; then
  ARMORY_BUILD_TYPE="local"
else
  ARMORY_BUILD_TYPE="prebuilt"
fi

echo "Building Images for Framework(s): ${FRAMEWORK[@]}"
for framework in "${FRAMEWORK[@]}"; do
  echo ""
  echo "------------------------------------------------"
  echo "Building docker image for framework: $framework"

  CMD="docker build"
  if $NO_CACHE; then
    CMD="$CMD --no-cache"
  else
    CMD="$CMD --cache-from ${REPO}/${framework}:${TAG}"
  fi


  CMD="$CMD --force-rm"
  CMD="$CMD --file $SCRIPT_DIR/Dockerfile-${framework}"
  if [ $framework == "base" ]; then
    CMD="$CMD --build-arg image_tag=${BASE_TAG}"
  else
    CMD="$CMD --build-arg image_tag=${TAG}"
  fi

  CMD="$CMD --build-arg armory_version=${ARMORY_VERSION}"

  if [ $framework != "base" ]; then
    if [ $ARMORY_BUILD_TYPE == "local" ]; then
      CMD="$CMD --target armory-local"
    else
      CMD="$CMD --target armory-prebuilt"
    fi
  fi


  CMD="$CMD "
  if [ $framework == "base" ]; then
    CMD="$CMD -t ${REPO}/${framework}:${BASE_TAG}"
  else
    CMD="$CMD -t ${REPO}/${framework}:${TAG}"
  fi
  CMD="$CMD $VERBOSE"
  CMD="$CMD ."

  if $DRYRUN; then
    echo "Would have Executed: "
    echo "    ->  $CMD"
  else
    echo "Executing: "
    echo "    ->  $CMD"
    $CMD
  fi

done

