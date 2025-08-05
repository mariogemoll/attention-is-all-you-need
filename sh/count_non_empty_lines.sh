#!/usr/bin/env bash

total=0
for f in "$@"; do
    # -c = count matches, -v = invert match, -e '^$' = empty lines
    # The LC_ALL speeds up grep by disabling locale processing.
    count=$(LC_ALL=C grep -cve '^[[:space:]]*$' -- "$f")
    printf "%7d %s\n" "$count" "$f"
    total=$((total + count))
done

if [ "$#" -gt 1 ]; then
    printf "%7d total\n" "$total"
fi
