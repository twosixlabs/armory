#!/usr/bin/env bash
echo "Executing 'json' formatter..."

TARGET_FILES=`echo ${TRACKED_FILES} | sed 's/ /\n/g' | grep -E '.*\.json$'`
[ -z "$TARGET_FILES" ] && exit 0


pushd $PROJECT_ROOT > /dev/null || exit 1
    for TARGET_FILE in ${TARGET_FILES}; do
        echo "Checking ${TARGET_FILE}..."
        python -mjson.tool --sort-keys --indent=4 ${TARGET_FILE} 2>&1 | diff ${TARGET_FILE} -
        if [ $? -ne 0 ] ; then
            JSON_PATCH="`python -mjson.tool --sort-keys --indent=4 ${TARGET_FILE}`"
            echo "${JSON_PATCH}" > $TARGET_FILE    # The double quotes are important here!
            echo "Lint check of JSON object failed. Your changes were not commited."
            echo "in ${PROJECT_ROOT}/${TARGET_FILE}"
            EXIT_STATUS=1
        fi
    done
popd
