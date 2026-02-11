#!/bin/bash
set -euo pipefail

predicted_dir=$1
output_dir=$2
truth_dir=$3
genome_file=$4


mkdir -p "$output_dir"

temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT

export predicted_dir truth_dir genome_file temp_dir output_dir

process_sample() {
    local sample="$1"
    local svtype="$2"

    # Get predicted CNV .bed and truth CNV .bed files for the sample and SV type
    predicted_file=$(ls "$predicted_dir"/${sample}.${svtype}.*.bed)
    truth_file=$(ls "$truth_dir"/${sample}/${sample}_${svtype}.bed)

    # Sort bed files by chromosome and start position, using genome file for correct sorting

    sorted_predicted=$temp_dir/"${sample}.${svtype}.predicted.sorted.bed"
    sorted_truth=$temp_dir/"${sample}.${svtype}.truth.sorted.bed"

    bedtools sort -i "$predicted_file" -g "$genome_file" > "$sorted_predicted"
    bedtools sort -i "$truth_file" -g "$genome_file" > "$sorted_truth"

    # Use bedtools intersect to find true positives (TP), false positives (FP), and false negatives (FN)
    tp_file=$output_dir/binary_classification/"${sample}.${svtype}.TP.bed"
    fp_file=$output_dir/binary_classification/"${sample}.${svtype}.FP.bed"
    fn_file=$output_dir/binary_classification/"${sample}.${svtype}.FN.bed"

    # True positives: predicted CNVs that overlap truth CNVs by at least 50%
    bedtools intersect \
        -a "$sorted_predicted" \
        -b "$sorted_truth" \
        -f 0.5 -r \
        -wa -wb \
        > "$tp_file"
    
    # False positives: predicted CNVs that do not overlap any truth CNVs by at least 50%
    bedtools intersect \
        -a "$sorted_predicted" \
        -b "$sorted_truth" \
        -f 0.5 -r \
        -v \
        -wa -wb \
        > "$fp_file"
    
    # False negatives: truth CNVs that do not overlap any predicted CNVs by at least 50%
    bedtools intersect \
        -a "$sorted_truth" \
        -b "$sorted_predicted" \
        -f 0.5 -r \
        -v \
        -wa -wb \
        > "$fn_file"    
}


# Get a list of sample names from the predicted directory (name is text before the first dot in the filename)
samples=$(ls "$predicted_dir"/*.*.bed | sed 's/.*\///; s/\..*//' | sort | uniq)
svtypes=("DEL" "DUP")

for sample in $samples; do
    for svtype in "${svtypes[@]}"; do
        process_sample "$sample" "$svtype"
    done
done

rm -rf "$temp_dir"