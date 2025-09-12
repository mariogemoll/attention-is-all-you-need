# Make script fail when a command fails
set -e

# Store directory the script is in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DOWNLOAD_DIR=$SCRIPT_DIR/../0_download

cd $DOWNLOAD_DIR

echo "Downloading files..."
URLS=(
    "https://www.statmt.org/europarl/v7/de-en.tgz"
    "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "https://www.statmt.org/wmt14/dev.tgz"
    "https://www.statmt.org/wmt14/test-full.tgz"
)

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
