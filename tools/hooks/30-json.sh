#!/usr/bin/env bash
echo "Executing 'black' formatter..."

pushd `git rev-parse --show-toplevel` > /dev/null

popd
