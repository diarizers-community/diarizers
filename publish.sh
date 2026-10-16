#!/bin/bash
set -o errexit

PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd "${PROJECT_PATH}"

rm -rf build
rm -rf dist
rm -rf src/*.egg-info diarizers.egg-info

python3 setup.py sdist bdist_wheel
python3 -m twine check dist/*
python3 -m twine upload dist/* --verbose

popd
