import os
import sys
import subprocess
import yaml
import json
import argparse
import re
import pandas as pd
from liftover import get_lifter
from typing import Optional
from collections import defaultdict
from utils import get_count_from_bed_file

from pathlib import Path
from cnv_parser import CNVParser
from benchmark_handler import BenchmarkParser

def _perform_liftover(config: dict, log_file: Optional[str | Path] = None):
    """
    Perform liftover on BED files based on configuration specifications.
    Reads coordinates from BED files, converts them using liftover, and overwrites files.
    
    Args:
        config: Configuration dictionary containing liftover specifications
    """

    results = defaultdict(dict)

    if 'liftover' not in config:
        print("No liftover specifications found in config. Skipping liftover.")
        return
    
    output_dir = Path(config['output_dir'])
    
    for dataset_name, liftover_spec in config['liftover'].items():
        print(f"Performing liftover for dataset: {dataset_name}")
        
        # Validate liftover specification
        from_build = liftover_spec['from']
        to_build = liftover_spec['to']

        if from_build == to_build:
            print(f"  Warning: 'from' and 'to' builds are the same ({from_build}). Skipping liftover for this dataset.")
            continue
        
        print(f"  Converting from {from_build} to {to_build}")

        # Log the liftover specifications for this dataset
        results[dataset_name]['from_build'] = from_build
        results[dataset_name]['to_build'] = to_build
        
        # Get the lifter
        converter = get_lifter(from_build, to_build, one_based=False)
        
        # Find the bed directory for this dataset
        dataset_subdir = dataset_name.replace(" ", "_")
        database_dir = output_dir / dataset_subdir
        
        if not database_dir.exists():
            print(f"  Warning: Database directory not found: {database_dir}")
            continue
        
        # Find all .bed files recursively
        bed_files = list(database_dir.glob("**/*.bed"))

        # Remove files in list from "binary_classification" subdirectories
        bed_files = [f for f in bed_files if "binary_classification" not in f.parts]
        
        if not bed_files:
            print(f"  Warning: No BED files found in {database_dir}")
            continue
        
        print(f"  Found {len(bed_files)} BED files to convert")

        # Initialize samples as a list instead of a dict
        results[dataset_name]['samples'] = []
        
        # First pass: collect all sample info
        sample_info = {}
        for bed_file in bed_files:
            count = get_count_from_bed_file(bed_file)
            sample_id = bed_file.stem  # Format: sample.svtype
            
            if sample_id not in sample_info:
                sample_info[sample_id] = {'record_count_before_liftover': count}
        
        total_converted = 0
        total_failed = 0
        total_size_failed = 0
        
        for bed_file in bed_files:
            # Check if file is empty
            if bed_file.stat().st_size == 0:
                print(f"    Skipping empty file: {bed_file.name}")
                continue

            # Read BED file (no header)
            df = pd.read_csv(bed_file, sep='\t', header=None)
            
            if df.empty:
                print(f"    Skipping empty file: {bed_file.name}")
                continue
            
            # Assuming standard BED format: chrom, start, end, [other columns...]
            df_cols = {0: 'chrom', 1: 'start_old', 2: 'end_old'}
            
            # Keep other columns with their original indices
            for i in range(3, len(df.columns)):
                df_cols[i] = f'col_{i}'
            df = df.rename(columns=df_cols)
            
            # Collect unique coordinates
            unique_coords = set()
            for idx, row in df.iterrows():
                chrom = row['chrom']
                unique_coords.add((chrom, row['start_old']))
                unique_coords.add((chrom, row['end_old']))
            
            # Perform liftover for each unique coordinate
            coord_map = {}
            failed_coords = 0
            for chrom, pos in unique_coords:
                try:
                    result = converter[chrom][pos]
                    if result:
                        new_pos = result[0][1]
                        coord_map[(chrom, pos)] = new_pos
                    else:
                        coord_map[(chrom, pos)] = None
                        failed_coords += 1
                except (IndexError, KeyError):
                    coord_map[(chrom, pos)] = None
                    failed_coords += 1
            
            # Map coordinates back to dataframe
            df['start'] = [coord_map.get((chrom, start)) for chrom, start in zip(df['chrom'], df['start_old'])]
            df['end'] = [coord_map.get((chrom, end)) for chrom, end in zip(df['chrom'], df['end_old'])]
            
            # Filter out rows with failed liftover
            original_count = len(df)
            df = df.dropna(subset=['start', 'end'])
            failed_count = original_count - len(df)
            
            # Filter out rows with >10% change in size
            size_threshold_pct = 0.10
            df['size_old'] = df['end_old'] - df['start_old']
            df['size_new'] = df['end'] - df['start']
            df['size_change'] = abs(df['size_new'] - df['size_old'])
            df['size_change_pct'] = df['size_change'] / df['size_old']
            df = df[df['size_change_pct'] <= size_threshold_pct]
            size_failed_count = original_count - len(df) - failed_count

            # Update totals
            total_converted += len(df)
            total_failed += failed_count
            total_size_failed += size_failed_count


            if failed_count > 0:
                print(f"    {bed_file.name}: {failed_count}/{original_count} records failed liftover")
            
            if size_failed_count > 0:
                print(f"    {bed_file.name}: {size_failed_count}/{original_count} records failed size change filter (>10% change)")
            
            if df.empty:
                print(f"    Warning: All records failed liftover for {bed_file.name}")
                continue
            
            # Convert to int
            df['start'] = df['start'].astype(int)
            df['end'] = df['end'].astype(int)
            
            # Reorder columns: chrom, start, end, then any additional columns
            output_cols = ['chrom', 'start', 'end']
            for col in df.columns:
                if col.startswith('col_'):
                    output_cols.append(col)
            df = df[output_cols]
            
            # Overwrite the original file
            df.to_csv(bed_file, sep='\t', index=False, header=False)

            # Extract sample and svtype from filename (format: sample.svtype)
            sample_id = bed_file.stem
            parts = sample_id.rsplit('.', 1)
            sample_name = parts[0] if len(parts) > 1 else sample_id
            svtype = parts[1] if len(parts) > 1 else "UNKNOWN"
            
            # Add record to samples list
            results[dataset_name]['samples'].append({
                'sample': sample_name,
                'svtype': svtype,
                'record_count_before_liftover': sample_info[sample_id]['record_count_before_liftover'],
                'record_count_after_liftover': len(df),
                'failed_liftover': failed_count,
                'failed_size_change': size_failed_count
            })
        
        print(f"  Liftover complete for {dataset_name}:")
        print(f"    Total records converted: {total_converted}")
        print(f"    Total records failed liftover: {total_failed}")
        print(f"    Total records failed size change filter: {total_size_failed}\n")
        
    # Save results to log file
    if log_file:
        os.makedirs(Path(log_file).parent, exist_ok=True)
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=4) 

def _parse_penncnv_to_bed(penncnv_file: str) -> pd.DataFrame:
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

def _convert_control_to_bed(config: dict):
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
            consensus_dir = output_dir / output_subdir_name / consensus_type
            if consensus_dir.exists():
                for file_path in consensus_dir.glob('*.bed'):
                    # Extract sample ID (string before first dot)
                    sample_id = file_path.stem.split('.')[0]
                    samples_of_interest.add(sample_id)
    
    if not samples_of_interest:
        print("Warning: No samples found in consensus calls directories. Skipping control processing.")
        return
    
    print(f"Found {len(samples_of_interest)} samples of interest from consensus calls directories")
    
    for control_name, control_path in config['control'].items():
        print(f"Processing control dataset: {control_name}")
        
        # Create output directory
        output_subdir_name = control_name.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name / "bed"
        os.makedirs(output_subdir, exist_ok=True)
        
        # Parse PennCNV file
        print(f"  Parsing PennCNV file: {control_path}")
        df = _parse_penncnv_to_bed(control_path)
        
        if df.empty:
            print(f"  Warning: No records found in {control_path}")
            continue
        
        # Filter to only samples of interest
        df = df[df['sample_id'].isin(samples_of_interest)]
        
        if df.empty:
            print(f"  Warning: No records found for samples of interest in {control_path}")
            continue
        
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

def _convert_vcfs_to_bed(config: dict):
    output_dir = Path(config['output_dir'])

    # For each input set, create a CNVParser instance and convert VCF files to BED format
    for key, input_map in config['input'].items():
        print(f"Converting input set: {key}")

        # Remove whitespace from key to create a valid directory name
        output_subdir_name = key.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name
        os.makedirs(output_subdir, exist_ok=True)

        # Create CNVParser instance and get all VCF files
        cnv_parser = CNVParser(input_map)
        all_vcf_files = cnv_parser.get_all_vcf_files()

        # Check that tool_patterns contains expected tools
        expected_tools = {"cnvpytor", "delly", "gatk"}
        if not expected_tools.issubset(set(cnv_parser.tool_patterns.keys())):
            print(f"Warning: Expected tools {expected_tools} not all found in tool patterns. Found: {set(cnv_parser.tool_patterns.keys())}")
            sys.exit(1)

        # Convert all VCF files and export to files
        for tool, id_path_pair in all_vcf_files.items():
            for sample_id, vcf_path in id_path_pair:
                data = cnv_parser.convert_vcf_to_bed(vcf_path)

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

def _run_consensus_calls_script(config: dict):
    output_dir = Path(config['output_dir'])
    results = {}

    for key, input_map in config['input'].items():
        print(f"Running consensus calls script for input set: {key}")
        # Remove whitespace from key to create a valid directory name
        output_subdir_name = key.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name

        command = [
            "./src/get_consensus_calls.sh",
            output_subdir / "bed/cnvpytor",
            output_subdir / "bed/delly",
            output_subdir / "bed/gatk",
            output_subdir,
            config['genome_file'],
            config['excluded_regions_file']
        ]

        subprocess.run(command, check=True)

        # Read log file into results dictionary, and remove after reading
        log_file = output_subdir / "get_consensus_calls_summary.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                results[key] = json.load(f)
        else:
            print(f"Warning: Log file not found for consensus calls script: {log_file}")
        os.remove(log_file)
    
    # Save results to file
    log_output_file = Path(config['output_dir']) / "logs" / "consensus_calls_results.json"
    os.makedirs(log_output_file.parent, exist_ok=True)
    with open(log_output_file, 'w') as f:
        json.dump(results, f, indent=4)

def _run_benchmark_processing_script(
        config: dict, 
        log_file: Optional[str | Path] = None
    ):
    
    benchmark_parser = BenchmarkParser(config['benchmark_map'])
    output_dir = Path(config['output_dir'])
    output_subdir = output_dir / "benchmark_parsing"
    os.makedirs(output_subdir, exist_ok=True)

    print("Parsing all benchmarks to BED format...")
    benchmark_parser.parse_all_benchmarks_to_bed(output_subdir, common_samples_only=True, genome_file_path=config['genome_file'])

    print("Merging parsed benchmarks across all benchmarks...")
    results_dict = benchmark_parser.merge_across_benchmarks(output_subdir, genome_file_path=config['genome_file'])
    
    # Save merged results to file
    if log_file:
        os.makedirs(Path(log_file).parent, exist_ok=True)
        with open(log_file, 'w') as f:
            json.dump(results_dict, f, indent=4)

def _run_binary_classification_script(config: dict):
    output_dir = Path(config['output_dir'])

    for key, input_map in config['input'].items():
        print(f"Running binary classification script for input set: {key}")
        # Remove whitespace from key to create a valid directory name
        output_subdir_name = key.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name

        command = [
            "./src/get_binary_classification.sh",
            output_subdir / "intersections",
            output_subdir / "binary_classification" / "intersections",
            output_dir / "benchmark_parsing" / "merged",
            config['genome_file']
        ]

        subprocess.run(command, check=True)

        command = [
            "./src/get_binary_classification.sh",
            output_subdir / "unions",
            output_subdir / "binary_classification" / "unions",
            output_dir / "benchmark_parsing" / "merged",
            config['genome_file']
        ]

        subprocess.run(command, check=True)
    
    for key, input_map in config['control'].items():
        print(f"Running binary classification script for control dataset: {key}")
        # Remove whitespace from key to create a valid directory name
        output_subdir_name = key.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name

        command = [
            "./src/get_binary_classification.sh",
            output_subdir / "bed",
            output_subdir / "binary_classification",
            output_dir / "benchmark_parsing" / "merged",
            config['genome_file']
        ]

        subprocess.run(command, check=True)

def main(config: dict):
    """
    Main computation pipeline.
    
    Args:
        config: Configuration dictionary loaded from YAML
    """
    print("\nStep 1: Converting VCF files to BED format...")
    _convert_vcfs_to_bed(config)

    print("\nStep 2: Running consensus calls script...")
    _run_consensus_calls_script(config)

    print("\nStep 3: Processing control datasets (SNP Array)...")
    _convert_control_to_bed(config)

    print("\nStep 4: Performing liftover on datasets (if configured)...")
    liftover_log = Path(config['output_dir']) / "logs" / "liftover_results.json"
    _perform_liftover(config, log_file=liftover_log)

    print("\nStep 5: Running benchmark processing script...")
    benchmark_log = Path(config['output_dir']) / "logs" / "benchmark_processing_results.json"
    _run_benchmark_processing_script(config, log_file=benchmark_log)

    print("\nStep 6: Running binary classification script...")
    _run_binary_classification_script(config)

    

if __name__ == "__main__":
    # Allow running standalone for testing
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='BlendedCNV Computation Pipeline')
    parser.add_argument('--config', '-c', required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)
