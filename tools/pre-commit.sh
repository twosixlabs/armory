#!/usr/bin/env bash

run_check () {
  echo "Running Check: $@"
  "$@" > /dev/null 2>&1
  need_format=$?
  if [ $need_format -ne 0 ]
  then
      "$@"
      echo Some Python files were formatted
      echo You need to run \`git add ...\`
      echo and then \`git commit ...\` again
      exit $need_format
  fi
}
#func(cmd):

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


#set -e
#echo "Performing Checks with black"
#python -m black --check ./ > /dev/null 2>&1
#need_format=$?
#if [ $need_format -ne 0 ]
#then
#    python -m black ./
#    echo Some Python files were formatted
#    echo You need to do git add and git commit again
#    exit $need_format
#fi
#
#echo "Performing Checks with tools.format_json"
#python -m tools.format_json --check > /dev/null 2>&1
#need_format=$?
#if [ $need_format -ne 0 ]
#then
#    python -m tools.format_json
#    echo "Some JSON files were formatted"
#    echo "You need to do git add and git commit again"
#    exit $need_format
#fi
#
#echo "Performing checks with yamllint"
#yamllint --no-warnings ./ > /dev/null 2>&1
#need_format=$?
#if [ $need_format -ne 0 ]
#then
#    yamllint --no-warnings ./
#    echo "Some YAML files were formatted"
#    echo "You need to do git add and git commit again"
#    exit $need_format
#fi
#
##echo "Performing checks with yamllint"
##yamllint --no-warnings ./
#
#echo "performing checks with flake8"
#python -m flake8 . >/dev/null 2>&1
#need_format=$?
#if [ $need_format -ne 0 ]
#then
#    yamllint --no-warnings ./
#    echo "Some YAML files were formatted"
#    echo "You need to do git add and git commit again"
#    exit $need_format
#fi
