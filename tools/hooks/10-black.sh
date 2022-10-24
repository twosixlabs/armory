#!/usr/bin/env bash
echo "Executing 'black' formatter..."

pushd "`git rev-parse --git-dir`/../" > /dev/null
  if black --check --diff --color .; then
    exit 0
  else
    python -m black ./
    # git add -u
    echo "Some files were formatted."
    # echo "You need to do git add and git commit again."
    exit 1
  fi
popd
