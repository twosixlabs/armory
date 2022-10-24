#!/usr/bin/env bash
# Copy or link this script into .git/hooks/
# It runs automatically in the project root directory (parent of .git/).

echo "Executing 'black' formatter..."

# python -m black --check ./ > /dev/null 2>&1
# need_format=$?
# set -e
# if [ $need_format -ne 0 ]
# then
#     python -m black ./
#     echo Some Python files were formatted
#     echo You need to do git add and git commit again
#     exit $need_format
# fi
# set +e