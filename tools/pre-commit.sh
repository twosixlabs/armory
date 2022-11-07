#!/usr/bin/env bash
# Copy or link this script into .git/hooks/
#   $ cp "`pwd`/tools/pre-commit.sh" "`pwd`/.git/hooks/pre-commit"
# This script runs automatically in the project's root directory (parent of .git/).

EXIT_STATUS=0
EXIT_EARLY=0
MAX_FILE_SIZE=32 # Max JSON file size in Megabytes.
PROJECT_ROOT=`git rev-parse --show-toplevel`

# Determine command to use in order to collect
# tracked files. Toggle between `dif` and `lstree`
# using the `ARMORY_CI_TEST` environmental variable.
# Example:
#   $ ARMORY_CI_TEST=1 ./tools/pre-commit.sh
ARMORY_CI_TEST="${ARMORY_COMMIT_HOOK_CI:-0}"

TRACKED_FILES="git --no-pager diff --name-only"
if [ "${ARMORY_CI_TEST}" -ne 0 ]; then
    TRACKED_FILES="git --no-pager ls-files"
fi
[ -z "$TRACKED_FILES" ] && exit 0


function CHECK_EXIT_STATUS ()
{
    if [ $1 -ne 0 ]; then
        EXIT_STATUS=1
        if [ $EXIT_EARLY -ne 0 ]; then
            echo "🚨 Pre-commit hooks failed. Please fix the issues and re-run 'git add' and 'git commit' 🚑"
            exit ${EXIT_STATUS}
        fi
    fi
}


pushd $PROJECT_ROOT > /dev/null || exit 1
    (
        TARGET_FILES=`${TRACKED_FILES} | grep -E '\.py$' | sed 's/\n/ /g'`
        [ -z "$TARGET_FILES" ] && exit 0

        ############
        # Black
        echo "⚫ Executing 'black' formatter..."
        python -m black --check > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            python -m black $TARGET_FILES
            echo "⚫ some files were formatted."
            CHECK_EXIT_STATUS 1
        fi

        ############
        # Flake8
        echo "🎱 Executing 'flake8' formatter..."
        python -m flake8 --config=.flake8 ${TARGET_FILES}
        CHECK_EXIT_STATUS $?
    )

    ############
    # JSON Linting
    echo "📄 Executing 'json' formatter..."
    TARGET_FILES=`${TRACKED_FILES} | sed 's/ /\n/g' | grep -E '.*\.json$'`
    for TARGET_FILE in ${TARGET_FILES}; do
        # Check if file is too large to be linted
        FILE_SIZE=`du -m ${TARGET_FILE} | cut -f1`
        if [ ${FILE_SIZE} -gt ${MAX_FILE_SIZE} ]; then
            echo "📄 Skipping ${TARGET_FILE} (too large)"
            continue
        fi
        python -mjson.tool --sort-keys --indent=4 ${TARGET_FILE} 2>&1 | diff - ${TARGET_FILE}
        if [ $? -ne 0 ] ; then
            JSON_PATCH="`python -mjson.tool --sort-keys --indent=4 ${TARGET_FILE}`"
            echo "${JSON_PATCH}" > ${TARGET_FILE}    # The double quotes are important here!
            echo "📄 modified ${PROJECT_ROOT}/${TARGET_FILE}"
            CHECK_EXIT_STATUS 1
        fi
    done
popd > /dev/null


if [ "${EXIT_STATUS}" -ne 0 ]; then
    echo "🚨 Pre-commit hooks failed. Please fix the issues and re-run 'git add' and 'git commit' 🚑"
fi


exit $EXIT_STATUS
