#!/usr/bin/env bash

echo "Performing Checks with black"
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

echo "Performing Checks with tools.format_json"
python -m tools.format_json --check > /dev/null 2>&1
need_format=$?
set -e
if [ $need_format -ne 0 ]
then
    python -m tools.format_json
    echo "Some JSON files were formatted"
    echo "You need to do git add and git commit again"
    exit $need_format
fi

#TODO Do we need the `if` cluases above for these below?
echo "Performing checks with yamllint"
yamllint --no-warnings ./

echo "performing checks with flake8"
python -m flake8 .
