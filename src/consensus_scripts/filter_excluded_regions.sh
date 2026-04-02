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