#!/usr/bin/env bash
echo "Executing 'yamllint' formatter..."

pushd `git rev-parse --show-toplevel` > /dev/null

  yamllint --no-warnings ./

  EXIT_STATUS=$?

popd
