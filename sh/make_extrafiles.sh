## Run the commands in the first section of notebook.ipynb to get the necessary source files before
## running this script!

# Make script fail when a command fails
set -e

# Show all commands
set -x

# Get absolute path for dst dir
DST_DIR="$( cd "$1" && pwd )"

# Expect dst dir to be empty
if [ "$(ls -A $DST_DIR)" ]; then
   echo "Destination directory $DST_DIR is not empty"
   exit 1
fi

# Store directory the script is in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $SCRIPT_DIR/../1_input

cp commoncrawl.de-en.annotation commoncrawl.de-en.de commoncrawl.de-en.en \
    europarl-v7.de-en.de europarl-v7.de-en.en \
    news-commentary-v9.de-en.de news-commentary-v9.de-en.en \
    newstest2013.de newstest2013.en \
    processed/newstest2014.de-en.de processed/newstest2014.de-en.en \
    processed/newstest2014.de-en.metadata.tsv \
    "$DST_DIR"

for FILE in $DST_DIR/*; do
    python $SCRIPT_DIR/../py/text_indexer.py "$FILE"
done

cp processed/newstest2014.de-en.metadatamapping.bin $DST_DIR

python $SCRIPT_DIR/../py/create_annotation_index.py $DST_DIR/commoncrawl.de-en.annotation $DST_DIR/commoncrawl.de-en.annidx
