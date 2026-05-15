#!/bin/bash
set -euo pipefail

# Parse arguments
tool_1_name=$1
tool_1_dir=$2
tool_2_name=$3
tool_2_dir=$4
tool_3_name=$5
tool_3_dir=$6
outdir=$7
genome_file=$8
excluded_regions_file="${9:-}"
reciprocal_threshold="${10:-0.5}"

mkdir -p "$outdir/intersections" "$outdir/unions"

export outdir tool_1_name tool_1_dir tool_2_name tool_2_dir tool_3_name tool_3_dir genome_file excluded_regions_file reciprocal_threshold

process_file() {
    local file="$1"
    local log_dir="$2"
    base_name=$(basename "$file")

    # Extract type (DEL or DUP) from filename
    if [[ "$base_name" =~ \.DEL\. ]]; then
        svtype="DEL"
    elif [[ "$base_name" =~ \.DUP\. ]]; then
        svtype="DUP"
    else
        svtype="UNKNOWN"
    fi

    # Derive sample name (everything before first dot)
    sample_name=$(echo "$base_name" | cut -d. -f1)

    # Create log file for this sample and SV type
    log_file="$log_dir/${sample_name}.${svtype}.json"

    tool_1_tmp=$(mktemp)
    tool_2_tmp=$(mktemp)
    tool_3_tmp=$(mktemp)

    # Copy first three columns to temporary files
    cut -f1-3 "$tool_1_dir/$base_name" > "$tool_1_tmp"
    cut -f1-3 "$tool_2_dir/$base_name" > "$tool_2_tmp"
    cut -f1-3 "$tool_3_dir/$base_name" > "$tool_3_tmp"

    three_way_intersect=$(mktemp)
    three_way_union=$(mktemp)

    # Find 3-way intersection (regions where all 3 tools agree)
    # Chain intersections: tool_1 ∩ tool_2 ∩ tool_3
    bedtools intersect -a "$tool_1_tmp" -b "$tool_2_tmp" -f "$reciprocal_threshold" -r | \
    bedtools intersect -a stdin -b "$tool_3_tmp" -f "$reciprocal_threshold" -r > "$three_way_intersect"

    # Find original regions from all 3 tools that participate in the 3-way overlap
    # Strategy: find which regions from each tool overlap with the 3-way intersection
    if [[ -s "$three_way_intersect" ]]; then
        {
            bedtools intersect -a "$tool_1_tmp" -b "$three_way_intersect" -u
            bedtools intersect -a "$tool_2_tmp" -b "$three_way_intersect" -u
            bedtools intersect -a "$tool_3_tmp" -b "$three_way_intersect" -u
        } > "$three_way_union"
    else
        # If no 3-way intersection, union file is empty
        touch "$three_way_union"
    fi

    rm "$tool_1_tmp" "$tool_2_tmp" "$tool_3_tmp"

    # Log the number of 3-way intersecting calls
    three_way_intersect_count=$(wc -l < "$three_way_intersect")
    three_way_union_count=$(wc -l < "$three_way_union")
    echo "{" > "$log_file"
    echo "  \"three_way_intersection\": {" >> "$log_file"
    echo "    \"intersection_count\": $three_way_intersect_count," >> "$log_file"
    echo "    \"union_count\": $three_way_union_count" >> "$log_file"
    echo "  }" >> "$log_file"

    intersection_file="$outdir/intersections/${base_name%.bed}.intersection.bed"
    union_file="$outdir/unions/${base_name%.bed}.union.bed"

    # Create tool string with all three tools
    all_tools="$tool_1_name|$tool_2_name|$tool_3_name"

    # Create intersection file - the overlapping regions where all 3 tools agree
    awk -v type="$svtype" -v tools="$all_tools" 'BEGIN{OFS="\t"} {print $1, $2, $3, type, tools}' \
        "$three_way_intersect" | \
    bedtools sort -i - -g "$genome_file" > "$intersection_file"

    # Create union file - all original regions from all 3 tools that participate
    union_pre_merge=$(mktemp)
    awk -v type="$svtype" -v tools="$all_tools" 'BEGIN{OFS="\t"} {print $1, $2, $3, type, tools}' \
        "$three_way_union" | \
    sort -k1,1 -k2,2n > "$union_pre_merge"

    if [[ -s "$union_pre_merge" ]]; then
        bedtools merge -c 4,5 -o distinct,distinct -i "$union_pre_merge" | \
        bedtools sort -i - -g "$genome_file" > "$union_file"
    else
        : > "$union_file"
    fi
    rm "$union_pre_merge"

    rm "$three_way_intersect" "$three_way_union"

    # Close log file with closing brace
    echo "}" >> "$log_file"
}

export -f process_file

# Determine number of cores to use for parallel processing (use 2/3 of available cores)
NCORES=$(nproc)
NCORES=$((NCORES * 2 / 3))
# echo "Processing DEL and DUP files in parallel using $NCORES cores..."

# Create log directory
log_dir="$outdir/logs"
mkdir -p "$log_dir"

# Process DEL and DUP files in parallel (using tool_1_dir as reference)
ls "$tool_1_dir"/*.DEL.bed "$tool_1_dir"/*.DUP.bed | \
parallel -j "$NCORES" process_file {} "$log_dir"

# Get unique sample names (everything before .DEL or .DUP)
samples=$(ls "$outdir/intersections"/*.intersection.bed | \
    sed 's/.*\///; s/\..*//' | \
    sort -u)

sample_count=$(echo "$samples" | wc -l)
# echo "Found $sample_count unique samples."

for sample in $samples; do
    # Combine intersection files
    intersection_del="$outdir/intersections/${sample}.DEL.intersection.bed"
    intersection_dup="$outdir/intersections/${sample}.DUP.intersection.bed"
    intersection_combined="$outdir/intersections/${sample}.intersection.bed"

    if [[ -f "$intersection_del" ]] && [[ -f "$intersection_dup" ]]; then
        cat "$intersection_del" "$intersection_dup" | \
        bedtools sort -i - -g "$genome_file" > "$intersection_combined"
    fi

    # Combine union files
    union_del="$outdir/unions/${sample}.DEL.union.bed"
    union_dup="$outdir/unions/${sample}.DUP.union.bed"
    union_combined="$outdir/unions/${sample}.union.bed"

    if [[ -f "$union_del" ]] && [[ -f "$union_dup" ]]; then
        cat "$union_del" "$union_dup" | \
        bedtools sort -i - -g "$genome_file" > "$union_combined"
    fi
done

# Concatenate all log files into a single json array
master_log_file="$outdir/get_consensus_3of3_calls_summary.json"
echo "[" > "$master_log_file"
log_files=$(ls "$log_dir"/*.json)
log_count=$(echo "$log_files" | wc -l)

for log_file in $log_files; do
    cat "$log_file" >> "$master_log_file"
    echo "," >> "$master_log_file"
done

# Remove the last comma and add closing bracket
sed -i '$ s/,$//' "$master_log_file"
echo "]" >> "$master_log_file"

# Remove individual log files
rm -r "$log_dir"
