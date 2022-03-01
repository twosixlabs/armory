#!/usr/bin/env bash

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
  echo ""
  echo "REQUIRED"
  echo "-f | --framework            Select Which framework to target:"
  echo "                                [all|base|tf2|pytorch|pytorch-deepspeech]"
  echo ""
  echo "OPTIONAL"
  echo "-av | --armory-version      Specify the armory version to install in image when "
  echo "                            framework != \`base\` (Note: base armory image does not"
  echo "                            contain armory "
  echo "                                (default: uses \`local\` scm version)"
  echo ""
  echo "-t | --tag                  Specify \`tag\` to use for the created image (e.g. \`latest\` in"
  echo "                            repo/image:latest.  If framework=all, this tag will be applied to "
  echo "                            all images.  "
  echo "                                (default: \`armory-version\`)"
  echo "-bt | --base-tag            Specify \`tag\` to use as the base image for frameworks != \`base\`"
  echo "                                (default: uses \`tag\`)"
  echo "                                Note: This will NOT tag the base image, it will simply "
  echo "                                specify the base_image to use for dervied images "
  echo "-nc | --no-cache            Build Images \`Clean\` (i.e. using --no-cache)"
  echo "-dr | --dry-run             Only show the Build calls (do not execute the builds)"
  echo "-v | --verbose              Show Logs in Plain Text (uses \`--progress=plain\`"
  echo "-r | --repo                 Set the Docker Repository (Default: \`twosixarmory\`"
  echo "-h | --help                 Print this Help."
  echo
  echo "----------------------------------------------------------------------------------------"
}

# Setting Defaults
POSITIONAL_ARGS=()
NO_CACHE=false
ARMORY_VERSION="$(python setup.py --version)"
BASE_ARMORY_VERSION="$ARMORY_VERSION"
DRYRUN=false
VERBOSE="--progress=auto"
REPO="twosixarmory"
FRAMEWORK=""
TAG=${ARMORY_VERSION%.*}
BASE_TAG=$TAG

# Making sure execution path is correct
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ "$PWD" != "$(dirname $SCRIPT_DIR)" ]; then
  echo "Must Execute build script from within armory/ folder that contains folder called \`docker\`"
  return 1
fi

## Parsing CLI Arguments

while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--framework)
      FRAMEWORK="$2"
      shift # past argument
      shift # past value
      ;;
    -av|--armory-version)
      ARMORY_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--tag)
      TAG="$2"
      shift # past argument
      shift # past value
      ;;
    -bt|--base-tag)
      BASE_TAG="$2"
      shift # past argument
      shift # past value
      ;;
    -nc|--no-cache)
      NO_CACHE=true
      shift # past argument
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
      exit 0
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

## Conducting Checks of CLI args

if [ "$FRAMEWORK" == "" ]; then
  echo "Must Specify Framework"
  exit 1
fi

if [[ "$FRAMEWORK" != "pytorch" &&  "$FRAMEWORK" != "pytorch-deepspeech" && "$FRAMEWORK" != "tf2" && "$FRAMEWORK" != "all"  && "$FRAMEWORK" != "base" ]]; then
    echo "ERROR: <framework> argument must be \`tf2\`, \`pytorch\`, \`pytorch-deepspeech\`, \`base\` or \`all\`, not \`$FRAMEWORK\`"
    exit 1
fi

if [ $FRAMEWORK == "all" ]; then
  echo "Setting Build to use all frameworks"
  FRAMEWORK=("base" "pytorch" "tf2" "pytorch-deepspeech")
else
  FRAMEWORK=($FRAMEWORK)
fi

if [ "$ARMORY_VERSION" == "$BASE_ARMORY_VERSION" ]; then
  ARMORY_BUILD_TYPE="local"
else
  ARMORY_BUILD_TYPE="prebuilt"
  TAG="$ARMORY_VERSION"
fi

echo "Reformatting Tag to remove + sign (if in there)"
TAG=${TAG/+/-}
BASE_TAG=${BASE_TAG/+/-}

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
  if [ $framework != "base" ]; then
    CMD="$CMD --build-arg base_image_tag=${BASE_TAG}"
    CMD="$CMD --build-arg armory_version=${ARMORY_VERSION}"
    if [ $ARMORY_BUILD_TYPE == "local" ]; then
      CMD="$CMD --target armory-local"
    else
      CMD="$CMD --target armory-prebuilt"
    fi
  fi

  CMD="$CMD -t ${REPO}/${framework}:${TAG}"
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

