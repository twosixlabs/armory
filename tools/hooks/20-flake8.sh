#!/usr/bin/env bash
echo "Executing 'flake8' formatter..."

pushd $PROJECT_ROOT > /dev/null
  python -m flake8        \
    --count               \
    --exit-zero           \
    --max-complexity=10   \
    --max-line-length=127 \
    --statistics          \
    --show-source         \
    --config=.flake8 ./
popd

