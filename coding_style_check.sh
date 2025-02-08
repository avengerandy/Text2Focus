#!/bin/bash

set -o errexit

echo "main code coding style check:"
pylint --recursive=y --rcfile=./src/.pylintrc --fail-under 10 ./src/sliding_window.py ./src/fitness.py ./src/pareto.py
black --check --diff ./src/sliding_window.py ./src/fitness.py ./src/pareto.py
isort --check --diff --profile black ./src/sliding_window.py ./src/fitness.py ./src/pareto.py

echo "test code coding style check:"
pylint --recursive=y --rcfile=./tests/.pylintrc --fail-under 10 ./tests/test_sliding_window.py ./tests/test_fitness.py ./tests/test_pareto.py
black --check --diff ./tests/test_sliding_window.py ./tests/test_fitness.py ./tests/test_pareto.py
isort --check --diff --profile black ./tests/test_sliding_window.py ./tests/test_fitness.py ./tests/test_pareto.py
