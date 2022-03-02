#!/usr/bin/env bash

Help()
{
  # Display Help
  echo "----------------------------------------------------------------------------------------"
  echo "                   Armory Docker Build Script                       "
  echo
  echo "This build script helps to build the docker images necessary for armory exectuion "
  echo "(in docker mode).  The primary purpose of this script is to build the images from "
  echo "scratch and to be used by the armory CI toolchain.  If modifications are necessary"
  echo "you can use \`--dry-run\` to get the \`docker build...\` commands directly and "
  echo "modify them as necessary. "
  echo ""
  echo "Armory uses scm versioning from git, therefore if you want to re-build containers "
  echo "from a stable armory release, simply checkout that tag and then run this script"
  echo ""
  echo "Syntax: build.sh [-f|nc|dr|v|h]"
  echo ""
  echo "REQUIRED"
  echo "-f | --framework            Select Which framework to target:"
  echo "                                [all|base|tf2|pytorch|pytorch-deepspeech]"
  echo ""
  echo "OPTIONAL"
  echo "-t | --tag                  Specify Additional Tag to apply to each image"
  echo "-nc | --no-cache            Build Images \`Clean\` (i.e. using --no-cache)"
  echo "-dr | --dry-run             Only show the Build calls (do not execute the builds)"
  echo "-v | --verbose              Show Logs in Plain Text (uses \`--progress=plain\`"
  echo "-h | --help                 Print this Help."
  echo
  echo "----------------------------------------------------------------------------------------"
}

get_tag_from_version ()
{
  local  input=$1
  arr=( ${input//\./ } ) # Split by .
  arr="${arr[@]:0:4}" # Keep 1st 4 elements
  arr="${arr// /.}" # Put it back together
  result="${arr/+/-}" # Replace + with -
  echo $result
}

# Setting Defaults
POSITIONAL_ARGS=()
NO_CACHE=false
ARMORY_VERSION="$(python setup.py --version)"
DRYRUN=false
VERBOSE="--progress=auto"
REPO="twosixarmory"
FRAMEWORK=""
TAG=$(get_tag_from_version $ARMORY_VERSION)
ADDITIONAL_TAG=""


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
    -t|--tag)
      ADDITIONAL_TAG="$2"
      shift # past argument
      shift # past argument
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
    CMD="$CMD --build-arg base_image_tag=${TAG}"
    CMD="$CMD --build-arg armory_version=${ARMORY_VERSION}"
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

  if [ "$ADDITIONAL_TAG" != "" ]; then
    if $DRYRUN; then
      echo "Would have Executed: "
      echo docker tag "${REPO}/${framework}:${TAG}" "${REPO}/${framework}:${ADDITIONAL_TAG}"
    else
      echo "Executing: "
      echo docker tag "${REPO}/${framework}:${TAG}" "${REPO}/${framework}:${ADDITIONAL_TAG}"
      docker tag "${REPO}/${framework}:${TAG}" "${REPO}/${framework}:${ADDITIONAL_TAG}"
    fi
  fi
done

