#!/usr/bin/env bash
echo "Executing 'black' formatter..."

cd `git rev-parse --show-toplevel` > /dev/null
  python -m black --check --diff --color {armory,tests,docker}/*.py
  if [ $? -ne 0 ]; then
    python -m black ./
    echo "Some files were formatted."
    echo "You need to do git add and git commit again."
    EXIT_STATUS=1
  fi
cd -
