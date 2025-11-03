#!/bin/bash

# Fail if a subcommand fails
set -e

# Print the commands
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR/../ts

npx eslint --fix src

cd $SCRIPT_DIR/../py

isort .

black --force-exclude 'seqs_pb2\.(py|pyi)$|\.ipynb$' .

nbqa isort *.ipynb

nbqa black --line-length=91 *.ipynb
