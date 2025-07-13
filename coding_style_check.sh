#!/bin/bash

set -o errexit

echo "main code coding style check:"
pylint --recursive=y --rcfile=./src/.pylintrc --fail-under 10 ./src ./experiment
black --check --diff ./src ./experiment ./owlv2 ./pyramid
isort --check --diff --profile black ./src ./experiment ./owlv2 ./pyramid

echo "test code coding style check:"
pylint --recursive=y --rcfile=./tests/.pylintrc --fail-under 10 ./tests
black --check --diff ./tests
isort --check --diff --profile black ./tests
