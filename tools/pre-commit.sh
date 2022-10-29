#!/usr/bin/env bash
# Copy or link this script into .git/hooks/
#   $ cp "`pwd`/tools/pre-commit.sh" "`pwd`/.git/hooks/pre-commit"
# This script runs automatically in the project's root directory (parent of .git/).

EXIT_STATUS=0
PROJECT_ROOT=`git rev-parse --show-toplevel`

# Environmental variable used to notify hooks we are running in a workflow.
ARMORY_CI_TEST="${ARMORY_CI_TEST:-0}"

# Collect tracked files based on ARMORY_CI_TEST
TRACKED_FILES="git diff HEAD --name-only"
if [ "${ARMORY_CI_TEST}" -ne 0 ]; then
    TRACKED_FILES="git ls-files"
fi


pushd $PROJECT_ROOT > /dev/null
    # Source hooks
    for FILE in `ls -1 ./tools/hooks/*.sh | sort`; do
        echo "Importing ${FILE}..."
        source "${FILE}"
        if [ $? -ne 0 ]; then
            EXIT_STATUS=1
        fi
    done
popd > /dev/null


if [ "${EXIT_STATUS}" -ne 0 ]; then
  echo "🚨 Pre-commit hooks failed. Please fix the issues and re-run 'git add' 🚑"
fi


exit $EXIT_STATUS
