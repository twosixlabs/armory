#!/usr/bin/env bash
echo "Executing 'black' formatter..."

TARGET_FILES=`echo ${TRACKED_FILES} | sed 's/ /\n/g' | grep -E '.*\.py$'`
[ -z "$TARGET_FILES" ] && exit 0


pushd $PROJECT_ROOT > /dev/null || exit 1
  python -m black --check --diff --color $TARGET_FILES
  if [ $? -ne 0 ]; then
    python -m black $TARGET_FILES
    echo "Some files were formatted."
    echo "You need to do git add and git commit again."
    EXIT_STATUS=1
  fi
popd
