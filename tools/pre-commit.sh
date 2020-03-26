#!/usr/bin/env bash

python -m black --check ./ > /dev/null 2>&1
need_format=$?

if [ $need_format -ne 0 ]
then
    python -m black ./
    echo Some Python files were formatted
    echo You need to do git add and git commit again
    exit $need_format
fi

python -m tools.format_json --check > /dev/null 2>&1
need_format=$?
if [ $need_format -ne 0 ]
then
    echo "Some JSON files need to be reformatted. Please run:"
    echo "python -m tools.format_json"
    exit $need_format
fi

set -e

yamllint --no-warnings ./

python -m flake8
