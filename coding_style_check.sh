#!/bin/bash

set -o errexit

echo "main code coding style check:"
pylint --recursive=y --rcfile=./src/.pylintrc --fail-under 10 ./src ./experiment
black --check --diff ./src ./experiment
isort --check --diff --profile black ./src ./experiment

echo "test code coding style check:"
pylint --recursive=y --rcfile=./tests/.pylintrc --fail-under 10 ./tests
black --check --diff ./tests
isort --check --diff --profile black ./tests
