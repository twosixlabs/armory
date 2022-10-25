#!/usr/bin/env bash
echo "Executing 'black' formatter..."

pushd `git rev-parse --show-toplevel` > /dev/null
  python -m black --check --diff --color .
  if [ $? -ne 0 ]; then
    python -m black ./
    echo "Some files were formatted."
    echo "You need to do git add and git commit again."
    EXIT_STATUS=1
  fi
popd
