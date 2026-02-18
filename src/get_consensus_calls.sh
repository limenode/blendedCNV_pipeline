#!/bin/bash
set -euo pipefail

cnvpytor_dir=$1
delly_dir=$2
gatk_dir=$3
outdir=$4
genome_file=$5
excluded_regions_file=$6


mkdir -p "$outdir/intersections" "$outdir/unions"

export outdir delly_dir cnvpytor_dir gatk_dir genome_file excluded_regions_file

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

    delly_tmp=$(mktemp)
    cnvpytor_tmp=$(mktemp)
    gatk_tmp=$(mktemp)

    # Copy first three columns to temporary files
    cut -f1-3 "$delly_dir/$base_name" > "$delly_tmp"
    cut -f1-3 "$cnvpytor_dir/$base_name" > "$cnvpytor_tmp"
    cut -f1-3 "$gatk_dir/$base_name" > "$gatk_tmp"

    # Log the number of calls before filtering (dictionary/json format)
    delly_count=$(wc -l < "$delly_tmp")
    cnvpytor_count=$(wc -l < "$cnvpytor_tmp")
    gatk_count=$(wc -l < "$gatk_tmp")
    echo "{" > "$log_file"
    echo "  \"sample\": \"$sample_name\"," >> "$log_file"
    echo "  \"svtype\": \"$svtype\"," >> "$log_file"
    echo "  \"before_excluded_regions\": {" >> "$log_file"
    echo "    \"delly\": $delly_count," >> "$log_file"
    echo "    \"cnvpytor\": $cnvpytor_count," >> "$log_file"
    echo "    \"gatk\": $gatk_count" >> "$log_file"
    echo "  }" >> "$log_file"

    # Sort temporary files and remove CNVs that are 50% or more in excluded regions
    for tmp in "$delly_tmp" "$cnvpytor_tmp" "$gatk_tmp"; do
        # First filter by chromosome names listed in the genome file
        # then sort the calls
        # then filter out calls that overlap excluded regions by 50% or more
        awk 'NR==FNR {chrom[$1]=1; next} ($1 in chrom)' "$genome_file" "$tmp" | \
        bedtools sort \
            -i - \
            -g "$genome_file" | \
        bedtools intersect \
            -a - \
            -b "$excluded_regions_file" \
            -v \
            -f 0.5 -r \
            -sorted \
            -g "$genome_file" > "$tmp.sorted.bed"
        
        mv "$tmp.sorted.bed" "$tmp"
    done

    # Log the number of calls after filtering
    delly_count_after=$(wc -l < "$delly_tmp")
    cnvpytor_count_after=$(wc -l < "$cnvpytor_tmp")
    gatk_count_after=$(wc -l < "$gatk_tmp")
    echo "  ,\"after_excluded_regions\": {" >> "$log_file"
    echo "    \"delly\": $delly_count_after," >> "$log_file"
    echo "    \"cnvpytor\": $cnvpytor_count_after," >> "$log_file"
    echo "    \"gatk\": $gatk_count_after" >> "$log_file"
    echo "  }" >> "$log_file"

    delly_cnvpytor_intersect=$(mktemp)
    delly_gatk_intersect=$(mktemp)
    cnvpytor_gatk_intersect=$(mktemp)

    delly_cnvpytor=$(mktemp)
    delly_gatk=$(mktemp)
    cnvpytor_gatk=$(mktemp)

    # Find overlaps for intersection (without -wa -wb to get only overlapping regions)
    bedtools intersect -a "$delly_tmp" -b "$cnvpytor_tmp" -f 0.5 -r > "$delly_cnvpytor_intersect"
    bedtools intersect -a "$delly_tmp" -b "$gatk_tmp" -f 0.5 -r > "$delly_gatk_intersect"
    bedtools intersect -a "$cnvpytor_tmp" -b "$gatk_tmp" -f 0.5 -r > "$cnvpytor_gatk_intersect"

    # Find overlaps and get original regions for union
    bedtools intersect -a "$delly_tmp" -b "$cnvpytor_tmp" -f 0.5 -r -wa -wb > "$delly_cnvpytor"
    bedtools intersect -a "$delly_tmp" -b "$gatk_tmp" -f 0.5 -r -wa -wb > "$delly_gatk"
    bedtools intersect -a "$cnvpytor_tmp" -b "$gatk_tmp" -f 0.5 -r -wa -wb > "$cnvpytor_gatk"

    rm "$delly_tmp" "$cnvpytor_tmp" "$gatk_tmp"

    # Log the number of intersecting calls for each pair of tools
    delly_cnvpytor_count=$(wc -l < "$delly_cnvpytor_intersect")
    delly_gatk_count=$(wc -l < "$delly_gatk_intersect")
    cnvpytor_gatk_count=$(wc -l < "$cnvpytor_gatk_intersect")
    echo "  ,\"intersections\": {" >> "$log_file"
    echo "    \"delly_cnvpytor\": $delly_cnvpytor_count," >> "$log_file"
    echo "    \"delly_gatk\": $delly_gatk_count," >> "$log_file"
    echo "    \"cnvpytor_gatk\": $cnvpytor_gatk_count" >> "$log_file"
    echo "  }" >> "$log_file"

    intersection_file="$outdir/intersections/${base_name%.bed}.intersection.bed"
    union_file="$outdir/unions/${base_name%.bed}.union.bed"

    # Intersection - only the overlapping regions
    {
        awk -v type="$svtype" 'BEGIN{OFS="\t"} {print $1, $2, $3, type, "cnvpytor,delly"}' "$delly_cnvpytor_intersect"
        awk -v type="$svtype" 'BEGIN{OFS="\t"} {print $1, $2, $3, type, "delly,gatk"}' "$delly_gatk_intersect"
        awk -v type="$svtype" 'BEGIN{OFS="\t"} {print $1, $2, $3, type, "cnvpytor,gatk"}' "$cnvpytor_gatk_intersect"
    } | \
    sort -k1,1 -k2,2n | \
    bedtools merge -c 4,5 -o distinct,distinct -i - > "$intersection_file"

    # Union - entire span of all calls that intersect
    {
        awk -v type="$svtype" 'BEGIN{OFS="\t"} {print $1, $2, $3, type, "cnvpytor,delly"}' "$delly_cnvpytor"
        awk -v type="$svtype" 'BEGIN{OFS="\t"} {print $4, $5, $6, type, "cnvpytor,delly"}' "$delly_cnvpytor"
        awk -v type="$svtype" 'BEGIN{OFS="\t"} {print $1, $2, $3, type, "delly,gatk"}' "$delly_gatk"
        awk -v type="$svtype" 'BEGIN{OFS="\t"} {print $4, $5, $6, type, "delly,gatk"}' "$delly_gatk"
        awk -v type="$svtype" 'BEGIN{OFS="\t"} {print $1, $2, $3, type, "cnvpytor,gatk"}' "$cnvpytor_gatk"
        awk -v type="$svtype" 'BEGIN{OFS="\t"} {print $4, $5, $6, type, "cnvpytor,gatk"}' "$cnvpytor_gatk"
    } | \
    sort -k1,1 -k2,2n | \
    bedtools merge -c 4,5 -o distinct,distinct -i - > "$union_file"

    rm "$delly_cnvpytor_intersect" "$delly_gatk_intersect" "$cnvpytor_gatk_intersect"
    rm "$delly_cnvpytor" "$delly_gatk" "$cnvpytor_gatk"

    # Function to deduplicate tool names in column 5
    deduplicate_tools() {
        local input_file="$1"
        local tmp_file=$(mktemp)

        awk 'BEGIN{OFS="\t"} {
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
        }' "$input_file" > "$tmp_file"

        mv "$tmp_file" "$input_file"
    }

    deduplicate_tools "$intersection_file"
    deduplicate_tools "$union_file"

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

# Process DEL and DUP files in parallel
ls "$delly_dir"/*.DEL.bed "$delly_dir"/*.DUP.bed | \
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
master_log_file="$outdir/get_consensus_calls_summary.json"
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
