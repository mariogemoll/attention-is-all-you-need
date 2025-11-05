#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ $# -ne 1 ]; then
  echo "Usage: $0 <label>"
  exit 1
fi

label="$1"
tmpfile="/tmp/${label}_sorted.csv"

echo "Processing ${label}..."

# Sort CSV alphabetically and store temporarily
sort "${label}.csv" -o "$tmpfile"

# Extract 2nd column (comma-delimited)
cut -d',' -f2 "$tmpfile" > "${label}.txt"

# Convert .txt → .bin
python $SCRIPT_DIR/../py/convert_numbers_txt2bin.py "${label}.txt" "${label}.bin"

# Clean up
rm -f "$tmpfile"

echo "✅ Done: ${label}.bin created (temporary files removed)."