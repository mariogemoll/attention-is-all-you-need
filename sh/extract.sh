#!/bin/bash

# Make script fail when a command fails
set -e

# Store directory the script is in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DOWNLOAD_DIR=$SCRIPT_DIR/../0_download
INPUT_DIR=$SCRIPT_DIR/../1_input

# List of all available datasets
ALL_DATASETS="europarl-v7 commoncrawl news-commentary-v9 newstest2013 newstest2014"

# Function to get extractor function name for a dataset
get_extractor() {
    case $1 in
        "europarl-v7") echo "extract_europarl_v7" ;;
        "commoncrawl") echo "extract_commoncrawl" ;;
        "news-commentary-v9") echo "extract_news_commentary_v9" ;;
        "newstest2013") echo "extract_newstest2013" ;;
        "newstest2014") echo "extract_newstest2014" ;;
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

# Create a temporary directory with a random name under /tmp
TMP_DIR=$(mktemp -d /tmp/XXXXXX)
echo "Temporary directory: " $TMP_DIR

cd $DOWNLOAD_DIR

# Generic function to extract datasets
# Usage: extract_dataset "dataset_name" "archive_file" "extracted_files..."
extract_dataset() {
    local dataset_name="$1"
    local archive_file="$2"
    shift 2
    local files_to_move=("$@")
    
    if [ -f "$archive_file" ]; then
        echo "Extracting $dataset_name..."
        tar -xzf "$archive_file" -C "$TMP_DIR"
        
        for file_path in "${files_to_move[@]}"; do
            local src_file="$TMP_DIR/$file_path"
            local filename=$(basename "$file_path")
            if [ -f "$src_file" ]; then
                mv "$src_file" "$INPUT_DIR/$filename"
            else
                echo "Warning: Expected file $src_file not found after extraction"
            fi
        done
    else
        echo "Warning: $archive_file not found, skipping $dataset_name"
    fi
}

# Dataset-specific extraction functions
extract_europarl_v7() {
    extract_dataset "europarl-v7" "de-en.tgz" \
        "europarl-v7.de-en.en" \
        "europarl-v7.de-en.de"
}

extract_commoncrawl() {
    extract_dataset "commoncrawl" "training-parallel-commoncrawl.tgz" \
        "commoncrawl.de-en.en" \
        "commoncrawl.de-en.de" \
        "commoncrawl.de-en.annotation"
}

extract_news_commentary_v9() {
    extract_dataset "news-commentary-v9" "training-parallel-nc-v9.tgz" \
        "training/news-commentary-v9.de-en.en" \
        "training/news-commentary-v9.de-en.de"
}

extract_newstest2013() {
    extract_dataset "newstest2013" "dev.tgz" \
        "dev/newstest2013.en" \
        "dev/newstest2013.de"
}

extract_newstest2014() {
    extract_dataset "newstest2014" "test-full.tgz" \
        "test-full/newstest2014-deen-src.en.sgm" \
        "test-full/newstest2014-deen-ref.de.sgm"
}

# Extract files based on dataset parameter
echo "Extracting files for dataset: $DATASET"

case $DATASET in
    "all")
        for dataset in $ALL_DATASETS; do
            extractor=$(get_extractor "$dataset")
            $extractor &
        done
        wait
        ;;
    *)
        extractor=$(get_extractor "$DATASET")
        if [[ -n "$extractor" ]]; then
            $extractor
        else
            echo "Error: Unknown dataset '$DATASET'"
            show_usage
        fi
        ;;
esac

echo "Deleting the temporary directory..."
rm -rf $TMP_DIR

echo "Done."