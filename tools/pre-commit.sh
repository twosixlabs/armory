#!/usr/bin/env bash
# Copy or link this script into .git/hooks/
# It runs automatically in the project root directory (parent of .git/).

PROJECT_ROOT=`git rev-parse --show-toplevel`


pushd $PROJECT_ROOT > /dev/null

    . ./tools/hooks/10-black.sh
    . ./tools/hooks/30-json.sh

    # TODO: Auto add files back?
    # git add -u
popd > /dev/null


# yamllint --no-warnings ./

# python -m flake8
