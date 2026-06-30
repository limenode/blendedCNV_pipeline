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

mkdir -p "$outdir/intersections" "$outdir/unions"

export outdir tool_1_name tool_1_dir tool_2_name tool_2_dir tool_3_name tool_3_dir genome_file

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
    echo "{" > "$log_file"
    echo "  \"all_calls\": {" >> "$log_file"
    echo "    \"total_count\": $all_calls_count" >> "$log_file"
    echo "  }" >> "$log_file"
    echo "}" >> "$log_file"

    intersection_file="$outdir/intersections/${base_name%.bed}.intersection.bed"
    union_file="$outdir/unions/${base_name%.bed}.union.bed"

    all_calls_with_id=$(mktemp)
    reciprocal_pairs=$(mktemp)
    pairwise_intersection_calls=$(mktemp)
    pairwise_union_calls=$(mktemp)
    merged_intersection_reciprocal=$(mktemp)
    merged_union_reciprocal=$(mktemp)
    singleton_non_reciprocal=$(mktemp)

    # Add a stable row id so we can track which calls participate in reciprocal overlaps.
    bedtools sort -i "$all_calls" -g "$genome_file" | \
    awk 'BEGIN{OFS="\t"} {print $1, $2, $3, $4, $5, NR}' > "$all_calls_with_id"

    # Build reciprocal-overlap call pairs (one direction only: id_a < id_b).
    bedtools intersect -a "$all_calls_with_id" -b "$all_calls_with_id" -r -f 0.5 -wa -wb | \
    awk 'BEGIN{OFS="\t"} $6 < $12 {
        inter_start = ($2 > $8 ? $2 : $8);
        inter_end = ($3 < $9 ? $3 : $9);
        union_start = ($2 < $8 ? $2 : $8);
        union_end = ($3 > $9 ? $3 : $9);
        print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, inter_start, inter_end, union_start, union_end
    }' > "$reciprocal_pairs"

    # Keep calls that are not part of any reciprocal pair as singletons.
    awk 'BEGIN{OFS="\t"} NR==FNR {paired[$6]=1; paired[$12]=1; next} !($6 in paired) {print $1, $2, $3, $4, $5}' \
        "$reciprocal_pairs" "$all_calls_with_id" > "$singleton_non_reciprocal"

    # Create pairwise reciprocal intersection and union intervals.
    awk 'BEGIN{OFS="\t"} {print $1, $13, $14, $4, $5 "|" $11}' "$reciprocal_pairs" > "$pairwise_intersection_calls"
    awk 'BEGIN{OFS="\t"} {print $1, $15, $16, $4, $5 "|" $11}' "$reciprocal_pairs" > "$pairwise_union_calls"

    # Merge reciprocal pairwise intersections.
    if [[ -s "$pairwise_intersection_calls" ]]; then
        bedtools sort -i "$pairwise_intersection_calls" -g "$genome_file" | \
        bedtools merge -c 4,5 -o distinct,collapse -i - | \
        awk 'BEGIN{OFS="\t"} {
            gsub(/\|/, ",", $5);
            split($5, tools, ",");
            delete seen;
            for (i in tools) {
                if (tools[i] != "") {
                    seen[tools[i]] = 1;
                }
            }

            n = asorti(seen, sorted_tools);
            result = "";
            for (i = 1; i <= n; i++) {
                result = result (i == 1 ? "" : "|") sorted_tools[i];
            }

            print $1, $2, $3, $4, result
        }' > "$merged_intersection_reciprocal"
    else
        : > "$merged_intersection_reciprocal"
    fi

    # Merge reciprocal pairwise unions.
    if [[ -s "$pairwise_union_calls" ]]; then
        bedtools sort -i "$pairwise_union_calls" -g "$genome_file" | \
        bedtools merge -c 4,5 -o distinct,collapse -i - | \
        awk 'BEGIN{OFS="\t"} {
            gsub(/\|/, ",", $5);
            split($5, tools, ",");
            delete seen;
            for (i in tools) {
                if (tools[i] != "") {
                    seen[tools[i]] = 1;
                }
            }

            n = asorti(seen, sorted_tools);
            result = "";
            for (i = 1; i <= n; i++) {
                result = result (i == 1 ? "" : "|") sorted_tools[i];
            }

            print $1, $2, $3, $4, result
        }' > "$merged_union_reciprocal"
    else
        : > "$merged_union_reciprocal"
    fi

    # Final outputs: reciprocal-derived calls + singleton calls.
    cat "$merged_intersection_reciprocal" "$singleton_non_reciprocal" | \
    bedtools sort -i - -g "$genome_file" > "$intersection_file"

    cat "$merged_union_reciprocal" "$singleton_non_reciprocal" | \
    bedtools sort -i - -g "$genome_file" > "$union_file"

    rm "$all_calls_with_id" "$reciprocal_pairs" "$pairwise_intersection_calls" "$pairwise_union_calls" "$merged_intersection_reciprocal" "$merged_union_reciprocal" "$singleton_non_reciprocal"

    rm "$all_calls"
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
master_log_file="$outdir/get_consensus_1of3_calls_summary.json"
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
