#!/usr/bin/env bash

ARMORY_ROOT_DIR=`git rev-parse --show-toplevel`
ARMORY_SRC_DIR="${ARMORY_ROOT_DIR}/armory"

# TODO: Style Guides(?)
#     - PEP8: https://peps.python.org/pep-0008/
#     - Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
python -m black \
  --config "${ARMORY_ROOT_DIR}/pyproject.toml" \
  --check \
  "${ARMORY_SRC_DIR}" > /dev/null 2>&1

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
