#!/usr/bin/env bash
# Build a docker image for specific framework. Optionally `all` frameworks can be built.
# Ex: `bash docker/build.sh pytorch`

if [ "$#" == 1 ]; then
    dev=""
elif [ "$#" == 2 ]; then
    if [[ "$2" != "--dev" && "$2" != "dev" ]]; then
        echo "ERROR: argument 2 needs to be \`dev\` or \`--dev\`, not \`$2\`"
        exit 1
    fi
    dev="-dev"
else
    echo "Usage: bash docker/build.sh <framework> [dev]"
    echo "    <framework> be \`tf1\`, \`tf2\`, \`pytorch\`, \`pytorch-deepspeech\`, or \`all\`"
    exit 1
fi


# Parse framework argument
if [[ "$1" == "armory" ]]; then
    # Deprecation. Remove for 0.14
    echo "ERROR: <framework> armory is on longer supported as of 0.13.0"
fi
if [[ "$1" != "pytorch" &&  "$1" != "pytorch-deepspeech" && "$1" != "tf1" && "$1" != "tf2" && "$1" != "all" ]]; then
    echo "ERROR: <framework> argument must be \`tf1\`, \`tf2\`, \`pytorch\`, \`pytorch-deepspeech\`, or \`all\`, not \`$1\`"
    exit 1
fi

# Parse Version
echo "Parsing Armory version"
version=$(python -m armory --version)
if [[ $version == *"-dev" ]]; then
    # Deprecation. Remove for 0.14
    echo "ERROR: Armory version $version ends in '-dev'. This is no longer supported as of 0.13.0"
    exit 1
fi
echo "Armory version $version"

# Build images
for framework in "tf1" "tf2" "pytorch" "pytorch-deepspeech"; do
    if [[ "$1" == "$framework" || "$1" == "all" ]]; then
        echo ""
        echo "------------------------------------------------"
        echo "Building docker image for framework: $framework"
        echo docker build --cache-from twosixarmory/${framework}:latest --force-rm --file docker/${framework}/Dockerfile --build-arg armory_version=${version} --target armory-${framework}${dev} -t twosixarmory/${framework}:${version} .
        docker build --cache-from twosixarmory/${framework}:latest --force-rm --file docker/${framework}/Dockerfile --build-arg armory_version=${version} --target armory-${framework}${dev} -t twosixarmory/${framework}:${version} .
        if [[ $dev == "-dev" ]]; then
            echo "Tagging docker image as latest"
            echo docker tag twosixarmory/${framework}:${version} twosixarmory/${framework}:latest
            docker tag twosixarmory/${framework}:${version} twosixarmory/${framework}:latest
        fi
    fi
done
