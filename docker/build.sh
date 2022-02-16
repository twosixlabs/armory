#!/usr/bin/env bash
# Build a docker image for specific framework. Optionally `all` frameworks can be built.
# Ex: `bash docker/build.sh pytorch`

POSITIONAL_ARGS=()
CLEAN=false
NO_CACHE=false
TAG="dev"
ARMORY_VERSION="local"

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
    --no-cache)
      NO_CACHE=true
      shift # past argument
      ;;
    -av|--armory-version)
      ARMORY_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
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
  FRAMEWORK=("base" "tf1" "tf2" "pytorch" "pytorch-deepspeech")
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
    CMD="$CMD --cache-from twosixarmory/armory-${framework}:${TAG}"
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
  CMD="$CMD -t twosixarmory/armory-${framework}:${TAG}"
  CMD="$CMD ."
  echo "->  $CMD"

done

## Build images
#for framework in "tf1" "tf2" "pytorch" "pytorch-deepspeech"; do
#    if [[ "$1" == "$framework" || "$1" == "all" ]]; then
#        echo ""
#        echo "------------------------------------------------"
#        echo "Building docker image for framework: $framework"
#        echo docker build --cache-from twosixarmory/${framework}:latest --force-rm --file docker/${framework}/Dockerfile --build-arg armory_version=${version} --target armory-${framework}${dev} -t twosixarmory/${framework}:${version} .
#        docker build --cache-from twosixarmory/${framework}:latest --force-rm --file docker/${framework}/Dockerfile --build-arg armory_version=${version} --target armory-${framework}${dev} -t twosixarmory/${framework}:${version} .
#        if [[ $dev == "-dev" ]]; then
#            echo "Tagging docker image as latest"
#            echo docker tag twosixarmory/${framework}:${version} twosixarmory/${framework}:latest
#            docker tag twosixarmory/${framework}:${version} twosixarmory/${framework}:latest
#        fi
#    fi
#done
#
#echo "HEllo"
#echo $FRAMEWORK
#echo $CLEAN
#echo $NO_CACHE

#if [ "$#" == 1 ]; then
#    dev=""
#elif [ "$#" == 2 ]; then
#    if [[ "$2" != "--dev" && "$2" != "dev" ]]; then
#        echo "ERROR: argument 2 needs to be \`dev\` or \`--dev\`, not \`$2\`"
#        exit 1
#    fi
#    dev="-dev"
#else
#    echo "Usage: bash docker/build.sh <framework> [dev]"
#    echo "    <framework> be \`tf1\`, \`tf2\`, \`pytorch\`, \`pytorch-deepspeech\`, or \`all\`"
#    exit 1
#fi
#
#
## Parse framework argument
#if [[ "$1" == "armory" ]]; then
#    # Deprecation. Remove for 0.14
#    echo "ERROR: <framework> armory is on longer supported as of 0.13.0"
#fi
#if [[ "$1" != "pytorch" &&  "$1" != "pytorch-deepspeech" && "$1" != "tf1" && "$1" != "tf2" && "$1" != "all" ]]; then
#    echo "ERROR: <framework> argument must be \`tf1\`, \`tf2\`, \`pytorch\`, \`pytorch-deepspeech\`, or \`all\`, not \`$1\`"
#    exit 1
#fi
#
## Parse Version
#echo "Parsing Armory version"
#version=$(python -m armory --version)
#if [[ $version == *"-dev" ]]; then
#    # Deprecation. Remove for 0.14
#    echo "ERROR: Armory version $version ends in '-dev'. This is no longer supported as of 0.13.0"
#    exit 1
#fi
#echo "Armory version $version"
#
## Build images
#for framework in "tf1" "tf2" "pytorch" "pytorch-deepspeech"; do
#    if [[ "$1" == "$framework" || "$1" == "all" ]]; then
#        echo ""
#        echo "------------------------------------------------"
#        echo "Building docker image for framework: $framework"
#        echo docker build --cache-from twosixarmory/${framework}:latest --force-rm --file docker/${framework}/Dockerfile --build-arg armory_version=${version} --target armory-${framework}${dev} -t twosixarmory/${framework}:${version} .
#        docker build --cache-from twosixarmory/${framework}:latest --force-rm --file docker/${framework}/Dockerfile --build-arg armory_version=${version} --target armory-${framework}${dev} -t twosixarmory/${framework}:${version} .
#        if [[ $dev == "-dev" ]]; then
#            echo "Tagging docker image as latest"
#            echo docker tag twosixarmory/${framework}:${version} twosixarmory/${framework}:latest
#            docker tag twosixarmory/${framework}:${version} twosixarmory/${framework}:latest
#        fi
#    fi
#done
