#!/bin/bash

# Script to count CNVs in BED files and output as JSON-like structure
# Usage: ./get_bed_counts.sh <directory> [lower_bound] [upper_bound]

# Check if directory argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <directory> [lower_bound] [upper_bound]" >&2
    echo "  directory: Path to search for .bed files" >&2
    echo "  lower_bound: Optional minimum CNV size for filtering" >&2
    echo "  upper_bound: Optional maximum CNV size for filtering" >&2
    exit 1
fi

DIRECTORY="$1"
LOWER_BOUND="${2:-}"
UPPER_BOUND="${3:-}"

# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist" >&2
    exit 1
fi

# Use associative array to store counts
declare -A counts

# Find all .bed files recursively
while IFS= read -r -d '' bed_file; do
    # Extract filename without path
    filename=$(basename "$bed_file")
    
    # Set sample_id as path without directory and extension
    sample_id="${bed_file#$DIRECTORY/}"
    sample_id="${sample_id%.bed}"
    
    # Count lines based on whether bounds are provided
    if [ -n "$LOWER_BOUND" ] && [ -n "$UPPER_BOUND" ]; then
        # Filter by size bounds using awk (size = end - start)
        count=$(awk -v lower="$LOWER_BOUND" -v upper="$UPPER_BOUND" '($3 - $2) >= lower && ($3 - $2) <= upper' "$bed_file" | wc -l)
    elif [ -n "$LOWER_BOUND" ]; then
        # Filter by lower bound only (size >= lower)
        count=$(awk -v lower="$LOWER_BOUND" '($3 - $2) >= lower' "$bed_file" | wc -l)
    elif [ -n "$UPPER_BOUND" ]; then
        # Filter by upper bound only (size <= upper)
        count=$(awk -v upper="$UPPER_BOUND" '($3 - $2) <= upper' "$bed_file" | wc -l)
    else
        # No filtering, just count lines
        count=$(wc -l < "$bed_file")
    fi
    
    # Add to counts (if sample_id already exists, add to it)
    if [ -n "${counts[$sample_id]}" ]; then
        counts[$sample_id]=$((counts[$sample_id] + count))
    else
        counts[$sample_id]=$count
    fi
done < <(find "$DIRECTORY" -type f -name "*.bed" -print0)

# Output JSON-like structure
echo "{"
first=true
for sample_id in $(printf '%s\n' "${!counts[@]}" | sort); do
    if [ "$first" = true ]; then
        first=false
    else
        echo ","
    fi
    echo -n "    \"$sample_id\": ${counts[$sample_id]}"
done
echo ""
echo "}"
