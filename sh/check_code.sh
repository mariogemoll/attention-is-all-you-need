#!/bin/bash

# Fail if a subcommand fails
set -e

# Print the commands
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR/../js
for notebook in ../py/*.ipynb; do
    npx check-notebook "$notebook" 
done

cd $SCRIPT_DIR/../ts
npx eslint -c basic.eslint.config.js basic.eslint.config.js eslint.config.js
npx eslint src
npx tsc

cd $SCRIPT_DIR/../py

flake8 --show-source *.py

isort --check-only --diff .

black --check --diff --force-exclude 'seqs_pb2\.(py|pyi)$|\.ipynb$' .

mypy --strict --pretty .

nbqa flake8 --show-source *.ipynb

nbqa isort --check-only --diff *.ipynb

nbqa black --check --diff --line-length=91 *.ipynb

nbqa mypy --strict --pretty *.ipynb
