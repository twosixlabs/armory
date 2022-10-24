#!/usr/bin/env bash

PROJECT_ROOT=`git rev-parse --git-dir`


pushd $PROJECT_ROOT/.. > /dev/null
    pushd tools > /dev/null

        . ./hooks/10-black.sh

    popd > /dev/null
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


