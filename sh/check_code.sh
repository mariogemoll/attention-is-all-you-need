#!/bin/bash

# Fail if a subcommand fails
set -e

# Print the commands
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR/../py

flake8 --show-source *.py

isort --check-only --diff *.py

black --check --diff *.py

mypy --strict --pretty *.py

cd $SCRIPT_DIR/../js
npx eslint
for notebook in ../*.ipynb; do
    node check_notebook_json.js "$notebook" 
done

cd $SCRIPT_DIR/..
nbqa flake8 --show-source *.ipynb

nbqa isort --check-only --diff *.ipynb

nbqa black --check --diff *.ipynb

nbqa mypy --strict --pretty *.ipynb
