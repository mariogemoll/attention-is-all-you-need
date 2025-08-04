# Make script fail when a command fails
set -e

# Store directory the script is in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOWNLOAD_DIR=$SCRIPT_DIR/../0_download

# Create a temporary directory with a random name under /tmp
TMP_DIR=$(mktemp -d /tmp/download.XXXXXX)
echo "Temporary directory: " $TMP_DIR
cd $TMP_DIR

echo "Downloading files..."
URLS=(
    "https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "https://www.statmt.org/wmt14/dev.tgz"
    "https://www.statmt.org/wmt14/test-full.tgz"
)

touch config.txt
for url in "${URLS[@]}"; do
    echo "url =" $url >> config.txt
    echo "output =" $(basename $url) >> config.txt
done
curl -K config.txt -Z

# Extract files
echo "Extracting files..."
tar -xzf training-parallel-europarl-v7.tgz &
tar -xzf training-parallel-commoncrawl.tgz &
tar -xzf training-parallel-nc-v9.tgz &
tar -xzf dev.tgz &
tar -xzf test-full.tgz &

wait

echo "Moving the relevant files to the download directory..."
mv $TMP_DIR/training/europarl-v7.de-en.en $DOWNLOAD_DIR
mv $TMP_DIR/training/europarl-v7.de-en.de $DOWNLOAD_DIR
mv $TMP_DIR/commoncrawl.de-en.en $DOWNLOAD_DIR
mv $TMP_DIR/commoncrawl.de-en.de $DOWNLOAD_DIR
mv $TMP_DIR/training/news-commentary-v9.de-en.en $DOWNLOAD_DIR
mv $TMP_DIR/training/news-commentary-v9.de-en.de $DOWNLOAD_DIR
mv $TMP_DIR/dev/newstest2013.en $DOWNLOAD_DIR
mv $TMP_DIR/dev/newstest2013.de $DOWNLOAD_DIR
mv $TMP_DIR/test-full/newstest2014-deen-ref.en.sgm $DOWNLOAD_DIR
mv $TMP_DIR/test-full/newstest2014-deen-ref.de.sgm $DOWNLOAD_DIR

echo "Deleting the temporary directory..."
rm -rf $TMP_DIR

echo "Done."
