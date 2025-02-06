#!/bin/bash

set -o errexit

echo "main code coding style check:"
pylint --recursive=y --rcfile=./src/.pylintrc --fail-under 10 ./src/sliding_window.py
black --check --diff ./src/sliding_window.py
isort --check --diff --profile black ./src/sliding_window.py

echo "test code coding style check:"
pylint --recursive=y --rcfile=./tests/.pylintrc --fail-under 10 ./tests/test_sliding_window.py
black --check --diff ./tests/test_sliding_window.py
isort --check --diff --profile black ./tests/test_sliding_window.py
