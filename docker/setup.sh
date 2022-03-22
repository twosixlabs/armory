#!/bin/bash

set -e

echo "Setting up things"
DEV_MODE=false
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --developer) DEV_MODE=true; shift ;;
        --) shift; break ;;
        *) POSITIONAL_ARGS+=("$1") ; shift ;;
    esac
done
echo "Things"
set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters
echo "Developer Mode: $DEV_MODE"
if $DEV_MODE; then
  echo "Installing Armory from mounted version"
  if [ ! -d "/host-armory-repo/" ]; then
    echo "ERROR!!:  Host Armory Repo is not mounted at: /host-armory-repo"
    exit 1
  fi
  echo "pip install now"
  pushd /host-armory-repo/
  /opt/conda/bin/pip install --use-feature=in-tree-build --no-cache-dir .
#  /opt/conda/bin/pip install --no-cache-dir .
  popd
else
  echo "Installing Armory from build version"
  pushd /build-armory-repo/
  armory_version=$(cat /build-armory-repo/build_version.txt)
  SETUPTOOLS_SCM_PRETEND_VERSION=${armory_version} /opt/conda/bin/pip install --use-feature=in-tree-build --no-cache-dir .
  popd

fi

echo "Configuring Armory"
armory configure --use-default

echo "Executing: $@"
exec "$@"