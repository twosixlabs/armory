#!/usr/bin/env bash
# Copy or link this script into .git/hooks/
# It runs automatically in the project root directory (parent of .git/).

echo "Executing 'black' formatter..."

if python -m black --check ./; then
  exit 0
else
  python -m black ./
  # git add -u
  echo "Some files were formatted."
  # echo "You need to do git add and git commit again."
  exit 1
fi
