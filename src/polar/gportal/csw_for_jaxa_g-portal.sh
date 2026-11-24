#!/bin/sh

BASE_URL="https://csw.gportal.jaxa.jp/csw"

## CSW Search Options ##
COUNT=3000 ## Max value is 3000##
START_TIME="2026-02-15T00:00:00" ## Search start date. yyyy-mm-ddThh:mm:ss ##
END_TIME="2026-02-28T23:59:59" ## Serch end date. yyyy-mm-ddThh:mm:ss ##
DATASET_ID="10002015" ## Please refer to https://img.gportal.jaxa.jp/upload/COMMON/G-Portal_datasetID_list.pdf ##
BBOX="-180,60,180,90" ## Lower-Longitude,Lower-Latitude,Upper-Longitude,Upper-Latitude ##

OUTPUT_FILE="results.xml" ## Output file name ##

#
# Validate BBOX
#

OLDIFS="${IFS}"
IFS=','
set -- ${BBOX}
IFS="${OLDIFS}"

MIN_LON="$1"
MIN_LAT="$2"
MAX_LON="$3"
MAX_LAT="$4"

# Check the number of elements

if [ $# -ne 4 ]; then
    echo "ERROR: Invalid BBOX setting."
    echo "Current BBOX = ${BBOX}"
    echo ""
    echo "BBOX must contain exactly four values:"
    echo "  min_lon,min_lat,max_lon,max_lat"
    exit 1
fi

if ! awk \
    -v minlon="${MIN_LON}" \
    -v minlat="${MIN_LAT}" \
    -v maxlon="${MAX_LON}" \
    -v maxlat="${MAX_LAT}" '
BEGIN {

    # Numeric check

    if (minlon !~ /^-?[0-9]+(\.[0-9]+)?$/) exit 1
    if (minlat !~ /^-?[0-9]+(\.[0-9]+)?$/) exit 1
    if (maxlon !~ /^-?[0-9]+(\.[0-9]+)?$/) exit 1
    if (maxlat !~ /^-?[0-9]+(\.[0-9]+)?$/) exit 1

    # Longitude range check

    if (minlon < -180 || minlon > 180) exit 1
    if (maxlon < -180 || maxlon > 180) exit 1

    # Latitude range check

    if (minlat < -90 || minlat > 90) exit 1
    if (maxlat < -90 || maxlat > 90) exit 1

    # Min / Max relationship check

    if (minlon > maxlon) exit 1
    if (minlat > maxlat) exit 1

    exit 0
}'; then

    echo "ERROR: Invalid BBOX setting."
    echo "Current BBOX = ${BBOX}"
    echo ""
    echo "Conditions:"
    echo "  - BBOX must contain four numeric values."
    echo "  - Minimum longitude : -180 to 180"
    echo "  - Maximum longitude : -180 to 180"
    echo "  - Minimum latitude  : -90 to 90"
    echo "  - Maximum latitude  : -90 to 90"
    echo "  - Minimum longitude must be less than or equal to maximum longitude."
    echo "  - Minimum latitude must be less than or equal to maximum latitude."

    exit 1
fi

#
# Initialize output file
#

: > "${OUTPUT_FILE}"

start_index=1
page=1
total_records=""

while :
do

    echo "================================="
    echo "Page       : ${page}"
    echo "StartIndex : ${start_index}"

    url="${BASE_URL}?service=CSW&version=3.0.0&request=GetRecords&outputFormat=application/xml&count=${COUNT}&startIndex=${start_index}&startTime=${START_TIME}&endTime=${END_TIME}&datasetId=${DATASET_ID}&bbox=${BBOX}"

    response=$(curl -k -s "${url}")

    if [ -z "${response}" ]; then
        echo "ERROR: Empty response"
        exit 1
    fi

    #
    # Get total number of matched records from the first response
    #

    if [ "${page}" -eq 1 ]; then

        total_records=$(printf '%s\n' "${response}" \
            | grep -o 'numberOfRecordsMatched="[0-9]*"' \
            | grep -o '[0-9]*')

        echo "Total Records : ${total_records}"

        if [ -z "${total_records}" ]; then
            echo "WARNING: numberOfRecordsMatched not found."
        fi
    fi

    #
    # Extract nextRecord value from the response
    #

    next_record=$(printf '%s\n' "${response}" \
        | grep -o 'nextRecord="[0-9]*"' \
        | head -1 \
        | grep -o '[0-9]*')

    echo "NextRecord : ${next_record}"

    #
    # Append the response to the output file
    #

    printf '%s\n\n' "${response}" >> "${OUTPUT_FILE}"

    #
    # Stop when nextRecord is 0
    #

    if [ -z "${next_record}" ] || [ "${next_record}" = "0" ]; then
        echo "Last page reached."
        break
    fi

    #
    # Use nextRecord as the next startIndex
    #

    start_index="${next_record}"

    page=$((page + 1))

done

echo "================================="
echo "Completed."
echo "Output File   : ${OUTPUT_FILE}"
echo "Total Records : ${total_records}"
echo "Pages Fetched : ${page}"
