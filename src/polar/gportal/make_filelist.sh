#!/bin/sh

INPUT_FILE="results.xml"
OUTPUT_FILE="files.lst"

grep "FILE:DOWNLOAD" "${INPUT_FILE}" \
| sed -n "s/.*'linkage':'\([^']*\)'.*/\1/p" \
| sed 's#^standard/#/product/Standard/#' \
| sed 's#^#get #' \
> "${OUTPUT_FILE}"

echo "Created ${OUTPUT_FILE}"
echo "Number of files: $(wc -l < "${OUTPUT_FILE}")"
