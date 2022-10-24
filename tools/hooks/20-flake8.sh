#!/usr/bin/env bash
echo "Executing 'flake8' formatter..."

pushd `git rev-parse --show-toplevel` > /dev/null

  python -m flake8        \
    --count               \
    --exit-zero           \
    --max-complexity=10   \
    --max-line-length=127 \
    --statistics          \
    --show-source         \
    --config=.flake8 ./

popd

# TODO: C901 - Determine if `# noqa: C901` is needed
#   python3 -m pip install astpretty
#   if flake8 code == C901:
#     astpretty(file) => save to artifacts
