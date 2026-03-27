from collections import defaultdict

import pandas as pd
from liftover import get_lifter

from pathlib import Path
import os
from cnv_parser import CNVParser
import re

def perform_liftover(
    input: pd.DataFrame, 
    from_build: str, 
    to_build: str,
    size_threshold_pct: float = 0.10,
) -> tuple[pd.DataFrame, list[dict]]:

    # If the input DataFrame doesn't have the required columns, raise an error
    required_cols = {'chrom', 'start', 'end'}
    if not required_cols.issubset(input.columns):
        raise ValueError(f"Input DataFrame must contain the following columns: {required_cols}")
    
    # If from_build and to_build are the same, return the input DataFrame as is
    if from_build == to_build:
        return input, [{
            'svtype': 'ALL',
            'record_count_before_liftover': len(input),
            'record_count_after_liftover': len(input),
            'failed_liftover': 0,
            'failed_size_change': 0,
        }]

    # Copy input DataFrame to avoid modifying the original
    input = input.copy()

    # Rename start and end columns to avoid confusion during liftover
    input.rename(columns={'start': 'start_old', 'end': 'end_old'}, inplace=True)
    
    # Add all unique (chrom, pos) pairs, where pos is either start_old or end_old
    unique_pos_set = set(
        zip(input['chrom'], input['start_old'])
    ) | set(
        zip(input['chrom'], input['end_old'])
    )

    # Get and use the liftover converter
    converter = get_lifter(from_build, to_build)
    coord_map = {}

    for chrom, pos in unique_pos_set:
        try:
            result = converter[chrom][pos]
            if result:
                coord_map[(chrom, pos)] = result[0][1]
            else:
                coord_map[(chrom, pos)] = None
        except (IndexError, KeyError):
            coord_map[(chrom, pos)] = None

    # Apply the coordinate mapping to the input DataFrame
    input['start'] = input.apply(lambda row: coord_map.get((row['chrom'], row['start_old'])), axis=1)
    input['end'] = input.apply(lambda row: coord_map.get((row['chrom'], row['end_old'])), axis=1)

    # Track whether each record successfully mapped both start and end positions
    input['map_succeeded'] = input['start'].notna() & input['end'].notna()

    # Calculate sizes and size changes for records that successfully mapped
    input['size_old'] = input['end_old'] - input['start_old']
    input['size_new'] = input['end'] - input['start']

    input['size_change'] = (input['size_new'] - input['size_old']).abs() if input['map_succeeded'].any() else 0
    input['size_change_pct'] = input['size_change'] / input['size_old'] if input['map_succeeded'].any() else 0

    # Track whether each record passes the size-change threshold after successful mapping
    input['below_size_change_threshold'] = input['map_succeeded'] & (input['size_change_pct'] <= size_threshold_pct)

    # Keep only records that pass both liftover checks
    output = input[input['map_succeeded'] & input['below_size_change_threshold']]

    # Log the results
    stats = []
    if 'svtype' in input.columns:
        for svtype in input['svtype'].unique():
            svtype_df = input[input['svtype'] == svtype]
            stats.append({
                'svtype': svtype,
                'record_count_before_liftover': len(svtype_df),
                'record_count_after_liftover': len(svtype_df[svtype_df['map_succeeded'] & svtype_df['below_size_change_threshold']]),
                'failed_liftover': int((~svtype_df['map_succeeded']).sum()),
                'failed_size_change': int((svtype_df['map_succeeded'] & ~svtype_df['below_size_change_threshold']).sum()),
            })

    stats.append({
        'svtype': 'ALL',
        'record_count_before_liftover': len(input),
        'record_count_after_liftover': len(output),
        'failed_liftover': int((~input['map_succeeded']).sum()),
        'failed_size_change': int((input['map_succeeded'] & ~input['below_size_change_threshold']).sum()),
    })

    # Convert start and end back to integers (after filtering out failed liftover records)
    output['start'] = output['start'].astype(int)
    output['end'] = output['end'].astype(int)

    return output, stats

def convert_vcfs_to_bed(config: dict) -> dict | None:
    output_dir = Path(config['output_dir'])
    liftover_stats = defaultdict(dict)

    # For each input set, create a CNVParser instance and convert VCF files to BED format
    for key, input_map in config['input'].items():
        print(f"Converting input set: {key}")

        do_liftover = config['liftover'].get(key) is not None
        if do_liftover:
            liftover_stats[key] = {
                'from': config['liftover'][key]['from'],
                'to': config['liftover'][key]['to'],
                'samples': {}
            }

        # Remove whitespace from key to create a valid directory name
        output_subdir_name = key.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name
        os.makedirs(output_subdir, exist_ok=True)

        # Create CNVParser instance and get all VCF files
        cnv_parser = CNVParser(input_map)
        all_vcf_files = cnv_parser.get_all_vcf_files()

        # Convert all VCF files and export to files
        for tool, id_path_pair in all_vcf_files.items():
            for sample_id, vcf_path in id_path_pair:
                data = cnv_parser.convert_vcf_to_bed(vcf_path)

                # Check if liftover is needed and perform it if necessary
                if do_liftover:
                    from_build = config['liftover'][key]['from']
                    to_build = config['liftover'][key]['to']

                    data, stats = perform_liftover(data, from_build, to_build)
                    liftover_stats[key]['samples'][sample_id] = stats

                # Export to file
                output_prefix = output_subdir / "bed" / tool / sample_id
                os.makedirs(output_prefix.parent, exist_ok=True)
                output_prefix = str(output_prefix)

                data[data["svtype"] == "DEL"].to_csv(
                    output_prefix + ".DEL.bed", sep="\t", index=False, header=False
                )
                data[data["svtype"] == "DUP"].to_csv(
                    output_prefix + ".DUP.bed", sep="\t", index=False, header=False
            )
        
    # return liftover stats
    return liftover_stats if liftover_stats else None

def convert_control_to_bed(config: dict) -> dict | None:
    """
    Convert control datasets (e.g., SNP Array from PennCNV) to BED format.
    Only performs BED conversion without consensus calls or further processing.
    Only processes samples that exist in the consensus calls directories.
    
    Args:
        config: Configuration dictionary containing control dataset paths
    """
    if 'control' not in config:
        print("No control datasets found in config. Skipping control processing.")
        return
    
    output_dir = Path(config['output_dir'])
    
    # Collect samples of interest from consensus calls directories
    samples_of_interest = set()
    for key in config['input'].keys():
        output_subdir_name = key.replace(" ", "_")
        
        # Check both intersections and unions directories
        for consensus_type in ['intersections', 'unions']:
            consensus_dir = output_dir / output_subdir_name / "consensus_2of3" / consensus_type
            if consensus_dir.exists():
                for file_path in consensus_dir.glob('*.bed'):
                    # Extract sample ID (string before first dot)
                    sample_id = file_path.stem.split('.')[0]
                    samples_of_interest.add(sample_id)
    
    if not samples_of_interest:
        print("Warning: No samples found in consensus calls directories. Skipping control processing.")
        return
    
    print(f"Found {len(samples_of_interest)} samples of interest from consensus calls directories")
    
    liftover_stats = defaultdict(dict)


    for control_name, control_path in config['control'].items():
        print(f"Processing control dataset: {control_name}")

        do_liftover = config['liftover'].get(control_name) is not None
        if do_liftover:
            liftover_stats[control_name] = {
                'from': config['liftover'][control_name]['from'],
                'to': config['liftover'][control_name]['to'],
                'samples': {}
            }
        
        # Create output directory
        output_subdir_name = control_name.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name / "bed"
        os.makedirs(output_subdir, exist_ok=True)
        
        # Parse PennCNV file
        print(f"  Parsing PennCNV file: {control_path}")
        df = parse_penncnv_to_bed(control_path)
        
        if df.empty:
            print(f"  Warning: No records found in {control_path}")
            continue
        
        # Filter to only samples of interest
        df = df[df['sample_id'].isin(samples_of_interest)]
        
        if df.empty:
            print(f"  Warning: No records found for samples of interest in {control_path}")
            continue

        # Perform liftover if configured for this control dataset
        if do_liftover:
            from_build = config['liftover'][control_name]['from']
            to_build = config['liftover'][control_name]['to']

            df, stats = perform_liftover(df, from_build, to_build)
            liftover_stats[control_name]['samples'] = stats
        
        print(f"  Found {len(df)} CNV records across {df['sample_id'].nunique()} samples (filtered to samples of interest)")
        
        # Group by sample and export to BED files
        for sample_id, sample_df in df.groupby('sample_id'):

            # Append source (output_subdir_name.lower()) to the dataframes
            sample_df['source'] = output_subdir_name.lower()

            # Export DEL and DUP separately
            del_df = sample_df[sample_df['svtype'] == 'DEL'][['chrom', 'start', 'end', 'svtype', 'source']]
            dup_df = sample_df[sample_df['svtype'] == 'DUP'][['chrom', 'start', 'end', 'svtype', 'source']]
            
            if not del_df.empty:
                del_output = output_subdir / f"{sample_id}.DEL.bed"
                del_df.to_csv(del_output, sep='\t', index=False, header=False)
                print(f"    Exported {len(del_df)} deletions for {sample_id}")
            
            if not dup_df.empty:
                dup_output = output_subdir / f"{sample_id}.DUP.bed"
                dup_df.to_csv(dup_output, sep='\t', index=False, header=False)
                print(f"    Exported {len(dup_df)} duplications for {sample_id}")
        
        print(f"  Control dataset '{control_name}' processing complete.\n")

    # Return liftover stats if any
    return liftover_stats if liftover_stats else None

def parse_penncnv_to_bed(penncnv_file: str) -> pd.DataFrame:
    """
    Parse PennCNV output format to BED format.
    
    PennCNV format example:
    chr3:191067244-191070300  numsnp=4  length=3,057  state1,cn=0 /path/to/HG00144.sig.tsv startsnp=rs9821594 endsnp=kgp17677268
    
    Args:
        penncnv_file: Path to PennCNV output file
    
    Returns:
        DataFrame with columns: chrom, start, end, svtype, sample_id
    """
    records = []
    
    with open(penncnv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
            
            # Parse chromosome and position: chr3:191067244-191070300
            pos_match = re.match(r'(chr[^:]+):(\d+)-(\d+)', parts[0])
            if not pos_match:
                continue
            
            chrom = pos_match.group(1)
            start = int(pos_match.group(2))
            end = int(pos_match.group(3))
            
            # Parse copy number: state1,cn=0
            cn = None
            for part in parts:
                if 'cn=' in part:
                    cn_match = re.search(r'cn=(\d+)', part)
                    if cn_match:
                        cn = int(cn_match.group(1))
                    break
            
            if cn is None:
                continue
            
            # Determine svtype based on copy number
            # cn=0 or cn=1 -> deletion, cn>=3 -> duplication, cn=2 -> normal (skip)
            if cn <= 1:
                svtype = 'DEL'
            elif cn >= 3:
                svtype = 'DUP'
            else:
                continue  # Skip normal copy number (cn=2)
            
            # Extract sample ID from file path (e.g., /path/to/HG00144.sig.tsv)
            sample_id = None
            for part in parts:
                if '.sig.tsv' in part or '.txt' in part:
                    # Extract sample ID from path
                    sample_match = re.search(r'([^/]+)\.sig\.tsv|([^/]+)\.txt', part)
                    if sample_match:
                        sample_id = sample_match.group(1) or sample_match.group(2)
                    break
            
            if not sample_id:
                # Try to extract from any path-like string
                for part in parts:
                    if '/' in part:
                        # Get basename without extension
                        basename = os.path.basename(part)
                        sample_id = os.path.splitext(basename)[0]
                        # Remove common suffixes
                        sample_id = sample_id.replace('.sig', '').replace('.bam', '')
                        break
            
            if not sample_id:
                print(f"Warning: Could not extract sample ID from line: {line}")
                continue
            
            records.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'svtype': svtype,
                'sample_id': sample_id
            })
    
    return pd.DataFrame(records)

