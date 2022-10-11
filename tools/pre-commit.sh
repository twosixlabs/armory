#!/usr/bin/env bash

python -m black --check ./ > /dev/null 2>&1
need_format=$?
set -e
if [ $need_format -ne 0 ]
then
    python -m black ./
    echo Some Python files were formatted
    echo You need to do git add and git commit again
    exit $need_format
fi
set +e

python -m tools.format_json --check > /dev/null 2>&1
need_format=$?
set -e
if [ $need_format -ne 0 ]
then
    python -m tools.format_json
    echo Some JSON files were formatted
    echo You need to do git add and git commit again
    exit $need_format
fi

yamllint --no-warnings ./

python -m flake8
