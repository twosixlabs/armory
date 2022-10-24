#!/usr/bin/env bash
# Copy or link this script into .git/hooks/
#   $ ln -s tools/pre-commit.sh .git/hooks/pre-commit
# It runs automatically in the project root directory (parent of .git/).

EXIT_STATUS=0
PROJECT_ROOT=`git rev-parse --show-toplevel`


pushd $PROJECT_ROOT > /dev/null
    for FILE in `ls -1 ./tools/hooks/*.sh | sort`; do
        echo "Importing ${FILE}..."
        . "${FILE}"
        if [ $? -ne 0 ]; then
            EXIT_STATUS=1
        fi
    done

    # TODO: Auto add files back?
    # git add -u
popd > /dev/null

if [ $EXIT_STATUS -ne 0 ]; then
  echo "🚨 Pre-commit hooks failed. Please run 'pre-commit run --all-files' locally to fix the issues 🚑"
fi

exit $EXIT_STATUS
