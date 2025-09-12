#!/bin/bash

# Make script fail when a command fails
set -e

# Store directory the script is in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DOWNLOAD_DIR=$SCRIPT_DIR/../0_download

# Define URLs for each dataset
declare -A DATASET_URLS=(
    ["europarl-v7"]="https://www.statmt.org/europarl/v7/de-en.tgz"
    ["commoncrawl"]="https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    ["news-commentary-v9"]="https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz"
    ["newstest2013"]="https://www.statmt.org/wmt14/dev.tgz"
    ["newstest2014"]="https://www.statmt.org/wmt14/test-full.tgz"
)

# Function to show usage
show_usage() {
    local datasets="all"
    for dataset in "${!DATASET_URLS[@]}"; do
        datasets="$datasets, $dataset"
    done
    echo "Usage: $0 <dataset>"
    echo "  dataset: $datasets"
    exit 1
}

# Check if parameter is provided
if [ $# -ne 1 ]; then
    show_usage
fi

DATASET=$1

cd $DOWNLOAD_DIR

echo "Downloading files for dataset: $DATASET"

# Determine which URLs to download
URLS=()
case $DATASET in
    "all")
        for dataset in "${!DATASET_URLS[@]}"; do
            URLS+=("${DATASET_URLS[$dataset]}")
        done
        ;;
    *)
        if [[ -n "${DATASET_URLS[$DATASET]}" ]]; then
            URLS=("${DATASET_URLS[$DATASET]}")
        else
            echo "Error: Unknown dataset '$DATASET'"
            show_usage
        fi
        ;;
esac

touch config.txt

# Only add to config.txt if file does not exist
for url in "${URLS[@]}"; do
    output_file=$(basename "$url")
    if [ ! -f "$output_file" ]; then
        echo "url = $url" >> config.txt
        echo "output = $output_file" >> config.txt
    else
        echo "$output_file already exists, skipping download."
    fi
done

# Only run curl if there are files to download
if [ -s config.txt ]; then
    curl -K config.txt -Z
else
    echo "All files already exist. No downloads needed."
fi

rm config.txt