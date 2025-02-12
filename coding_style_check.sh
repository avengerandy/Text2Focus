#!/bin/bash

set -o errexit

echo "main code coding style check:"
pylint --recursive=y --rcfile=./src/.pylintrc --fail-under 10 ./src
black --check --diff ./src
isort --check --diff --profile black ./src

echo "test code coding style check:"
pylint --recursive=y --rcfile=./tests/.pylintrc --fail-under 10 ./tests
black --check --diff ./tests/test_sliding_window.py ./tests
isort --check --diff --profile black ./tests/test_sliding_window.py ./tests
