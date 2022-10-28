#!/usr/bin/env bash
echo "Executing 'json' formatter..."

PROJECT_ROOT=`git rev-parse --show-toplevel`

pushd $PROJECT_ROOT > /dev/null
    for FILE in `git diff HEAD --name-only | grep -P '\.json'`; do
        python -mjson.tool $FILE 2>&1 /dev/null
        if [ $? -ne 0 ] ; then
            echo "Lint check of JSON object failed. Your changes were not commited."
            echo "in ${PROJECT_ROOT}/${FILE}:"
            python -mjson.tool "${FILE}"
            EXIT_STATUS=1
        fi
    done
popd
