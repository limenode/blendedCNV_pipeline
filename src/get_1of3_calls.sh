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

mkdir -p "$outdir/intersections" "$outdir/unions"

export outdir tool_1_name tool_1_dir tool_2_name tool_2_dir tool_3_name tool_3_dir genome_file excluded_regions_file

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

    # Log the number of calls before filtering (dictionary/json format)
    tool_1_count=$(wc -l < "$tool_1_tmp")
    tool_2_count=$(wc -l < "$tool_2_tmp")
    tool_3_count=$(wc -l < "$tool_3_tmp")
    echo "{" > "$log_file"
    echo "  \"sample\": \"$sample_name\"," >> "$log_file"
    echo "  \"svtype\": \"$svtype\"," >> "$log_file"
    echo "  \"before_excluded_regions\": {" >> "$log_file"
    echo "    \"$tool_1_name\": $tool_1_count," >> "$log_file"
    echo "    \"$tool_2_name\": $tool_2_count," >> "$log_file"
    echo "    \"$tool_3_name\": $tool_3_count" >> "$log_file"
    echo "  }" >> "$log_file"

    # Sort temporary files and optionally remove CNVs that are 50% or more in excluded regions
    for tmp in "$tool_1_tmp" "$tool_2_tmp" "$tool_3_tmp"; do
        # First filter by chromosome names listed in the genome file and sort the calls
        awk 'NR==FNR {chrom[$1]=1; next} ($1 in chrom)' "$genome_file" "$tmp" | \
        bedtools sort \
            -i - \
            -g "$genome_file" > "$tmp.sorted.bed"
        
        # Only filter by excluded regions if the file exists and is not empty
        # Also skip if placeholder values like "-" or "none" are passed
        if [[ -n "$excluded_regions_file" && "$excluded_regions_file" != "-" && "$excluded_regions_file" != "none" && -f "$excluded_regions_file" && -s "$excluded_regions_file" ]]; then
            bedtools intersect \
                -a "$tmp.sorted.bed" \
                -b "$excluded_regions_file" \
                -v \
                -f 0.5 \
                -sorted \
                -g "$genome_file" > "$tmp.filtered.bed"
            mv "$tmp.filtered.bed" "$tmp.sorted.bed"
        fi
        
        mv "$tmp.sorted.bed" "$tmp"
    done

    # Log the number of calls after filtering
    tool_1_count_after=$(wc -l < "$tool_1_tmp")
    tool_2_count_after=$(wc -l < "$tool_2_tmp")
    tool_3_count_after=$(wc -l < "$tool_3_tmp")
    echo "  ,\"after_excluded_regions\": {" >> "$log_file"
    echo "    \"$tool_1_name\": $tool_1_count_after," >> "$log_file"
    echo "    \"$tool_2_name\": $tool_2_count_after," >> "$log_file"
    echo "    \"$tool_3_name\": $tool_3_count_after" >> "$log_file"
    echo "  }" >> "$log_file"

    all_calls=$(mktemp)

    # Combine all calls from all three tools (1/3 consensus - union approach)
    {
        awk -v type="$svtype" -v tool="$tool_1_name" 'BEGIN{OFS="\t"} {print $1, $2, $3, type, tool}' "$tool_1_tmp"
        awk -v type="$svtype" -v tool="$tool_2_name" 'BEGIN{OFS="\t"} {print $1, $2, $3, type, tool}' "$tool_2_tmp"
        awk -v type="$svtype" -v tool="$tool_3_name" 'BEGIN{OFS="\t"} {print $1, $2, $3, type, tool}' "$tool_3_tmp"
    } | sort -k1,1 -k2,2n > "$all_calls"

    rm "$tool_1_tmp" "$tool_2_tmp" "$tool_3_tmp"

    # Count total calls from all tools
    all_calls_count=$(wc -l < "$all_calls")
    echo "  ,\"all_calls\": {" >> "$log_file"
    echo "    \"total_count\": $all_calls_count" >> "$log_file"
    echo "  }" >> "$log_file"

    intersection_file="$outdir/intersections/${base_name%.bed}.intersection.bed"
    union_file="$outdir/unions/${base_name%.bed}.union.bed"

    # Create intersection file - all individual calls with their source tool
    bedtools sort -i "$all_calls" -g "$genome_file" > "$intersection_file"

    # Create union file - merge overlapping regions and concatenate tool names
    bedtools merge -c 4,5 -o distinct,collapse -i "$all_calls" | \
    awk 'BEGIN{OFS="\t"} {
        # Deduplicate and sort tool names
        split($5, tools, ",");
        delete seen;
        result = "";
        for (i in tools) {
            if (!(tools[i] in seen)) {
                seen[tools[i]] = 1;
                result = result (result == "" ? "" : "|") tools[i];
            }
        }
        print $1, $2, $3, $4, result
    }' | bedtools sort -i - -g "$genome_file" > "$union_file"

    rm "$all_calls"

    # Close log file with closing brace
    echo "}" >> "$log_file"
}

export -f process_file

# Determine number of cores to use for parallel processing (use 2/3 of available cores)
NCORES=$(nproc)
NCORES=$((NCORES * 2 / 3))
echo "Processing DEL and DUP files in parallel using $NCORES cores..."

# Create log directory
log_dir="$outdir/logs"
mkdir -p "$log_dir"

# Process DEL and DUP files in parallel (using tool_1_dir as reference)
ls "$tool_1_dir"/*.DEL.bed "$tool_1_dir"/*.DUP.bed | \
parallel -j "$NCORES" process_file {} "$log_dir"

# Post-processing: combine DEL and DUP files for each sample
echo "Combining DEL and DUP files for each sample..."

# Get unique sample names (everything before .DEL or .DUP)
samples=$(ls "$outdir/intersections"/*.intersection.bed | \
    sed 's/.*\///; s/\..*//' | \
    sort -u)

sample_count=$(echo "$samples" | wc -l)
echo "Found $sample_count unique samples."

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
master_log_file="$outdir/get_consensus_1of3_calls_summary.json"
echo "[" > "$master_log_file"
log_files=$(ls "$log_dir"/*.json)
log_count=$(echo "$log_files" | wc -l)
echo "Found $log_count log files to summarize."
for log_file in $log_files; do
    cat "$log_file" >> "$master_log_file"
    echo "," >> "$master_log_file"
done

# Remove the last comma and add closing bracket
sed -i '$ s/,$//' "$master_log_file"
echo "]" >> "$master_log_file"

# Remove individual log files
rm -r "$log_dir"

echo "Done!"
