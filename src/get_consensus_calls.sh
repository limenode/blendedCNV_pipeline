#!/bin/bash
set -euo pipefail

cnvpytor_dir=$1
delly_dir=$2
gatk_dir=$3

outdir=$4

# 5th argument (optional): expected number of samples (for assertion)
expected_samples=${5:-}

mkdir -p "$outdir/intersections" "$outdir/unions"

export outdir delly_dir cnvpytor_dir gatk_dir

process_file() {
    local file="$1"
    base_name=$(basename "$file")

    # Extract type (DEL or DUP) from filename
    if [[ "$base_name" =~ \.DEL\. ]]; then
        svtype="DEL"
    elif [[ "$base_name" =~ \.DUP\. ]]; then
        svtype="DUP"
    else
        svtype="UNKNOWN"
    fi
    
    delly_tmp=$(mktemp)
    cnvpytor_tmp=$(mktemp)
    gatk_tmp=$(mktemp)

    # Copy first three columns to temporary files
    cut -f1-3 "$delly_dir/$base_name" > "$delly_tmp"
    cut -f1-3 "$cnvpytor_dir/$base_name" > "$cnvpytor_tmp"
    cut -f1-3 "$gatk_dir/$base_name" > "$gatk_tmp"

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

}


export -f process_file
NCORES=64

echo "Processing DEL and DUP files in parallel using $NCORES cores..."

ls "$delly_dir"/*.DEL.bed "$delly_dir"/*.DUP.bed | \
parallel -j "$NCORES" process_file {}

# Post-processing: combine DEL and DUP files for each sample
echo "Combining DEL and DUP files for each sample..."

# Get unique sample names (everything before .DEL or .DUP)
samples=$(ls "$outdir/intersections"/*.intersection.bed | \
    sed 's/.*\///; s/\.DEL\..*//; s/\.DUP\..*//' | \
    sort -u)

# Assert expected number of samples found (if provided)
if [[ -n "$expected_samples" ]]; then
    actual_sample_count=$(echo "$samples" | wc -l)
    if [[ "$actual_sample_count" -ne "$expected_samples" ]]; then
        echo "Error: Expected $expected_samples samples, but found $actual_sample_count."
        exit 1
    fi
fi

for sample in $samples; do
    # Combine intersection files
    intersection_del="$outdir/intersections/${sample}.DEL.intersection.bed"
    intersection_dup="$outdir/intersections/${sample}.DUP.intersection.bed"
    intersection_combined="$outdir/intersections/${sample}.intersection.bed"
    
    if [[ -f "$intersection_del" ]] && [[ -f "$intersection_dup" ]]; then
        cat "$intersection_del" "$intersection_dup" | \
        sort -k1,1 -k2,2n > "$intersection_combined"
        echo "Created $intersection_combined"
    fi
    
    # Combine union files
    union_del="$outdir/unions/${sample}.DEL.union.bed"
    union_dup="$outdir/unions/${sample}.DUP.union.bed"
    union_combined="$outdir/unions/${sample}.union.bed"
    
    if [[ -f "$union_del" ]] && [[ -f "$union_dup" ]]; then
        cat "$union_del" "$union_dup" | \
        sort -k1,1 -k2,2n > "$union_combined"
        echo "Created $union_combined"
    fi
done

echo "Done!"