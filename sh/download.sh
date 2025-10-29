#!/bin/bash

# Make script fail when a command fails
set -e

# Store directory the script is in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DOWNLOAD_DIR=$SCRIPT_DIR/../0_download

# List of all available datasets
ALL_DATASETS="europarl-v7 commoncrawl news-commentary-v9 newstest2013 newstest2014"

# Function to get URL for a dataset
get_dataset_url() {
    case $1 in
        "europarl-v7") echo "https://www.statmt.org/europarl/v7/de-en.tgz" ;;
        "commoncrawl") echo "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz" ;;
        "news-commentary-v9") echo "https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz" ;;
        "newstest2013") echo "https://www.statmt.org/wmt14/dev.tgz" ;;
        "newstest2014") echo "https://www.statmt.org/wmt14/test-full.tgz" ;;
        *) echo "" ;;
    esac
}

# Function to show usage
show_usage() {
    echo "Usage: $0 <dataset>"
    echo "  dataset: all, $ALL_DATASETS"
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
        for dataset in $ALL_DATASETS; do
            url=$(get_dataset_url "$dataset")
            URLS+=("$url")
        done
        ;;
    *)
        url=$(get_dataset_url "$DATASET")
        if [[ -n "$url" ]]; then
            URLS=("$url")
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