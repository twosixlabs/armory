#!/usr/bin/env bash
# Copy or link this script into .git/hooks/
# It runs automatically in the project root directory (parent of .git/).

PROJECT_ROOT=`git rev-parse --show-toplevel`


pushd $PROJECT_ROOT > /dev/null

    . ./tools/hooks/10-black.sh

popd > /dev/null


# python -m tools.format_json --check > /dev/null 2>&1
# need_format=$?
# set -e
# if [ $need_format -ne 0 ]
# then
#     python -m tools.format_json
#     echo Some JSON files were formatted
#     echo You need to do git add and git commit again
#     exit $need_format
# fi

# yamllint --no-warnings ./

# python -m flake8


