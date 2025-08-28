# Make script fail when a command fails
set -e

# Store directory the script is in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DOWNLOAD_DIR=$SCRIPT_DIR/../0_download
INPUT_DIR=$SCRIPT_DIR/../1_input

# Create a temporary directory with a random name under /tmp
TMP_DIR=$(mktemp -d /tmp/XXXXXX)
echo "Temporary directory: " $TMP_DIR


cd $DOWNLOAD_DIR

# Extract files
echo "Extracting files..."
tar -xzf training-parallel-europarl-v7.tgz -C $TMP_DIR &
tar -xzf training-parallel-commoncrawl.tgz -C $TMP_DIR &
tar -xzf training-parallel-nc-v9.tgz -C $TMP_DIR &
tar -xzf dev.tgz -C $TMP_DIR &
tar -xzf test-full.tgz -C $TMP_DIR &

wait

echo "Moving the relevant files to the input directory..."
mv $TMP_DIR/training/europarl-v7.de-en.en $INPUT_DIR
mv $TMP_DIR/training/europarl-v7.de-en.de $INPUT_DIR
mv $TMP_DIR/commoncrawl.de-en.en $INPUT_DIR
mv $TMP_DIR/commoncrawl.de-en.de $INPUT_DIR
mv $TMP_DIR/commoncrawl.de-en.annotation $INPUT_DIR
mv $TMP_DIR/training/news-commentary-v9.de-en.en $INPUT_DIR
mv $TMP_DIR/training/news-commentary-v9.de-en.de $INPUT_DIR
mv $TMP_DIR/dev/newstest2013.en $INPUT_DIR
mv $TMP_DIR/dev/newstest2013.de $INPUT_DIR
mv $TMP_DIR/test-full/newstest2014-deen-ref.en.sgm $INPUT_DIR
mv $TMP_DIR/test-full/newstest2014-deen-ref.de.sgm $INPUT_DIR

echo "Deleting the temporary directory..."
rm -rf $TMP_DIR

echo "Done."