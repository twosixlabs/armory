#!/usr/bin/env bash

set -e

run_check () {
  echo "Running Check: $@"
  "$@" > /dev/null
  echo "Result"
  need_format=$?
  echo "Need Format: $need_format"
  if [ $need_format -ne 0 ]
  then
      exec "$@"
      echo Some Python files were formatted
      echo You need to run \`git add ...\`
      echo and then \`git commit ...\` again
      echo "Exiting with $need_format"
      exit $need_format
  fi
}

declare -a cmds=(
  "python -m black ./" \
  "python -m tools.format_json" \
  "yamllint --no-warnings ./" \
  "python -m flake8 ."
)

for i in "${cmds[@]}"
do
   run_check $i
   # do whatever on "$i" here
done

echo "All Checks completed succesfully!!"

