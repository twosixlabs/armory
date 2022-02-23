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
  echo "                                [all|base|tf1|tf2|pytorch|pytorch-deepspeech]"
  echo "-t | --tag                  Tag to use when creating Docker Image (e.g. \`latest\` in"
  echo "                                repo/image:latest"
  echo "-nc | --no-cache            Build Images \"Clean\" (i.e. using --no-cache)"
  echo "-av | --armory-version      Select armory version to install in container"
  echo "                              (default: local)"
  echo "-dr | --dry-run             Only show the Build calls (do not execute the builds)"
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

    -h|--help)
      Help
      exit
      ;;
    -*|--*)
      echo "Unknown option $1"
      echo "For more info try: build.sh -h"
      exit 1
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
  exit 1
fi

if [[ "$FRAMEWORK" != "pytorch" &&  "$FRAMEWORK" != "pytorch-deepspeech" && "$FRAMEWORK" != "tf1" && "$FRAMEWORK" != "tf2" && "$FRAMEWORK" != "all"  && "$FRAMEWORK" != "base" ]]; then
    echo "ERROR: <framework> argument must be \`tf1\`, \`tf2\`, \`pytorch\`, \`pytorch-deepspeech\`, \`base\` or \`all\`, not \`$FRAMEWORK\`"
    exit 1
fi

if [ $FRAMEWORK == "all" ]; then
  echo "setting all frameowks"
  FRAMEWORK=("base" "pytorch" "tf1" "tf2" "pytorch-deepspeech")
fi

# Parse Version
internal_armory_version=$(python -m armory --version)
if [[ $internal_armory_version == *"-dev" ]]; then
    # Deprecation. Remove for 0.14
    echo "ERROR: Armory version $internal_armory_version ends in '-dev'. This is no longer supported as of 0.13.0"
    exit 1
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
  CMD="$CMD --file docker/Dockerfile-${framework}"
  CMD="$CMD --build-arg image_tag=${TAG}"
  CMD="$CMD --build-arg armory_version=${ARMORY_VERSION}"

  if [ $framework != "base" ]; then
    if [ $ARMORY_BUILD_TYPE == "local" ]; then
      CMD="$CMD --target armory-local"
    else
      CMD="$CMD --target armory-prebuilt"
    fi
  fi


  CMD="$CMD "
  CMD="$CMD -t ${REPO}/${framework}:${TAG}"
  CMD="$CMD $VERBOSE"
  CMD="$CMD ."

  if [ $framework == "tf1" ]; then
    echo "Framework tf1 is deprecated....please use tf2"
  else
    if $DRYRUN; then
      echo "Would have Executed: "
      echo "    ->  $CMD"
    else
      echo "Executing: "
      echo "    ->  $CMD"
      $CMD

    fi
  fi
done

