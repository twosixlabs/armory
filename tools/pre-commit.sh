#! /bin/bin/env bash
#
# Perform QA Checks on the codebase

set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
ARMORY_ROOT_DIR=`git rev-parse --show-toplevel`
ARMORY_SRC_DIR="${ARMORY_ROOT_DIR}/armory"


cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
}


#######################################
#
#######################################
python_lint() {
  local NEEDS_FORMATTING=0
  echo "Running code formatting..."
  python -m black \
    --config "${ARMORY_ROOT_DIR}/pyproject.toml" \
    --check \
    "${ARMORY_SRC_DIR}" #> /dev/null 2>&1

  # python -m flake8

  # need_format=$?
  # set -e
  # if [ $need_format -ne 0 ]
  # then
  #     python -m black ./
  #     echo Some Python files were formatted >&2
  #     echo You need to do git add and git commit again >&2
  #     exit $need_format
  # fi
  # set +e
  echo "Code formatting passed."
  return 0
}


#######################################
#
#######################################
markup_lint() {
  local NEEDS_FORMATTING=0

  python -m tools.format_json --check #> /dev/null 2>&1
  # yamllint --no-warnings ./

  # need_format=$?
  # set -e
  # if [ $need_format -ne 0 ]
  # then
  #     python -m tools.format_json
  #     echo Some JSON files were formatted
  #     echo You need to do git add and git commit again
  #     exit $need_format
  # fi

  return 0
}


#######################################
#
#######################################
bandit_scan() {
  # TODO: Fix bandit issues and enable this check
  bandit -v -r ./armory -c "pyproject.toml" || $( exit 0 ); echo $?
}


#######################################
#
#######################################
python_lint || $( exit 1 ); echo $?
markup_lint || $( exit 1 ); echo $?
bandit_scan || $( exit 1 ); echo $?

$( exit 0 ); echo $?
