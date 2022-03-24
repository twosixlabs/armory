#!/usr/bin/env bash

Help()
{
  # Display Help
  echo "----------------------------------------------------------------------------------------"
  echo "                   Armory Docker Registry Update Script                                "
  echo
  echo "This script deploys the armory docker images to dockerhub.  Note: generally it is best "
  echo "practice to build the containers first"
  echo
  echo "Syntax: update_registry.sh <image1> <image2> ..."
  echo
  echo "options:"
  echo "-dr | --dry-run             Only show the Build calls (do not execute the builds)"
  echo "-h | --help                 Print this Help."
  echo
  echo "----------------------------------------------------------------------------------------"
}

DRYRUN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      Help
      exit 0
      ;;
    -dr|--dry-run)
      DRYRUN=true
      shift # past argument
      ;;

    -*|--*)
      echo "Unknown option $1"
      echo "For more info try: update_registry.sh -h"
      return 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

echo "Pushing the following to dockerhub: ${POSITIONAL_ARGS[@]} "

if [ "${POSITIONAL_ARGS}" = "" ]; then
  echo "Must specify at least one image"
  exit 1
fi

for image in "${POSITIONAL_ARGS[@]}"; do
  echo ""
  echo "------------------------------------------------"
  echo "Pushing Image: $image"
  CMD="echo not yet....soon to come"
  if $DRYRUN; then
    echo "Would have Executed: "
    echo "    ->  $CMD"
  else
    echo "Executing: "
    echo "    ->  $CMD"
#    $CMD
  fi
done

