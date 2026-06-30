from collections import defaultdict
from pathlib import Path
import os
import re
from cyvcf2 import VCF
from typing import Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor
import subprocess
from io import StringIO
import pandas as pd
from pybedtools import BedTool

from liftover import get_lifter
from utils import ensure_chr_prefix, sanitize_svtype
from cnv_parser import CNVParser
from output_layout import OutputLayout

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

    # Drop intermediate columns used for liftover checks
    output.drop(columns=['start_old', 'end_old', 'map_succeeded', 'size_old', 'size_new', 'size_change', 'size_change_pct', 'below_size_change_threshold'], inplace=True)

    return output, stats

def excluded_regions_filter(input_bed_path: str, excluded_bed_path: str):
    input_bed = BedTool(input_bed_path)
    excluded_bed = BedTool(excluded_bed_path)

    # Perform bedtools intersect -v with 50% overlap
    filtered_bed = input_bed.intersect(excluded_bed, v=True, f=0.5)

    # Save the filtered BED over the original file
    filtered_bed.saveas(input_bed_path)

def _process_single_vcf(
    vcf_path: str,
    tool: str,
    sample_id: str,
    input_map: dict,
    valid_chromosomes,
    output_prefix: str,
    do_liftover: bool,
    liftover_from: str | None,
    liftover_to: str | None,
    excluded_regions_file: str | None,
) -> list | None:
    cnv_parser = CNVParser(input_map)
    data = cnv_parser.convert_vcf_to_bed(vcf_path, source=tool, valid_chromosomes=valid_chromosomes)

    stats = None
    if do_liftover and liftover_from and liftover_to:
        data, stats = perform_liftover(data, liftover_from, liftover_to)

    data = data[['chrom', 'start', 'end', 'svtype', 'source']]
    data[data["svtype"] == "DEL"].to_csv(output_prefix + ".DEL.bed", sep="\t", index=False, header=False)
    data[data["svtype"] == "DUP"].to_csv(output_prefix + ".DUP.bed", sep="\t", index=False, header=False)

    if excluded_regions_file:
        excluded_regions_filter(output_prefix + ".DEL.bed", excluded_regions_file)
        excluded_regions_filter(output_prefix + ".DUP.bed", excluded_regions_file)

    return stats

def parse_vcfs_to_bed(config: dict) -> dict | None:
    layout = OutputLayout(Path(config['output_dir']))
    liftover_stats = defaultdict(dict)
    futures_to_meta = {}

    cpu_count = os.cpu_count()
    target_workers = max(1, (2 * cpu_count) // 3) if cpu_count else 1

    with ProcessPoolExecutor(max_workers=target_workers) as executor:
        for key, input_map in config['input'].items():
            print(f"Converting input set: {key}")

            do_liftover = config['liftover'].get(key) is not None
            if do_liftover:
                liftover_stats[key] = {
                    'from': config['liftover'][key]['from'],
                    'to': config['liftover'][key]['to'],
                    'samples': {}
                }

            output_subdir = layout.set_dir(key)
            os.makedirs(output_subdir, exist_ok=True)

            cnv_parser = CNVParser(input_map)
            all_vcf_files = cnv_parser.get_all_vcf_files()
            valid_chromosomes = config.get('valid_chromosomes', None)

            liftover_from = config['liftover'][key]['from'] if do_liftover else None
            liftover_to = config['liftover'][key]['to'] if do_liftover else None
            excluded_regions_file = config.get('excluded_regions_file') or None

            for tool, id_path_pair in all_vcf_files.items():
                for sample_id, vcf_path in id_path_pair:
                    output_prefix = str(layout.bed_dir(key, tool) / sample_id)
                    os.makedirs(Path(output_prefix).parent, exist_ok=True)

                    future = executor.submit(
                        _process_single_vcf,
                        vcf_path, tool, sample_id, input_map, valid_chromosomes,
                        output_prefix, do_liftover, liftover_from, liftover_to, excluded_regions_file,
                    )
                    futures_to_meta[future] = (key, sample_id)

        for future, (key, sample_id) in futures_to_meta.items():
            stats = future.result()
            if stats is not None:
                liftover_stats[key]['samples'][sample_id] = stats

    return liftover_stats if liftover_stats else None

def parse_control_to_bed(config: dict) -> dict | None:
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

    layout = OutputLayout(Path(config['output_dir']))

    # Collect samples of interest from consensus calls directories
    samples_of_interest = set()
    for key in config['input'].keys():
        # Check both intersections and unions directories
        for consensus_type in ['intersections', 'unions']:
            consensus_dir = layout.consensus_rep_dir(key, 2, consensus_type)
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
        output_subdir = layout.control_bed_dir(control_name)
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

        # Limit to chromosomes in config['valid_chromosomes'] if available
        if 'valid_chromosomes' in config:
            df = df[df['chrom'].isin(config['valid_chromosomes'])]

        # Perform liftover if configured for this control dataset
        if do_liftover:
            from_build = config['liftover'][control_name]['from']
            to_build = config['liftover'][control_name]['to']

            df, stats = perform_liftover(df, from_build, to_build)
            liftover_stats[control_name]['samples'] = stats
        
        print(f"  Found {len(df)} CNV records across {df['sample_id'].nunique()} samples (filtered to samples of interest)")
        
        # Group by sample and export to BED files
        for sample_id, sample_df in df.groupby('sample_id'):

            # Append source (sanitized control name, lowercased) to the dataframes
            sample_df['source'] = control_name.replace(" ", "_").lower()

            # Export DEL and DUP separately
            del_df = sample_df[sample_df['svtype'] == 'DEL'][['chrom', 'start', 'end', 'svtype', 'source']]
            dup_df = sample_df[sample_df['svtype'] == 'DUP'][['chrom', 'start', 'end', 'svtype', 'source']]
            
            if not del_df.empty:
                del_output = output_subdir / f"{sample_id}.DEL.bed"
                del_df.to_csv(del_output, sep='\t', index=False, header=False)
                # print(f"    Exported {len(del_df)} deletions for {sample_id}")
            
            if not dup_df.empty:
                dup_output = output_subdir / f"{sample_id}.DUP.bed"
                dup_df.to_csv(dup_output, sep='\t', index=False, header=False)
                # print(f"    Exported {len(dup_df)} duplications for {sample_id}")
        
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
            start = int(pos_match.group(2)) - 1  # Convert to 0-based
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

def _merge_one_sample(
        sample_id: str, 
        svtype: str, 
        sample_df: pd.DataFrame, 
        genome_file_path: str,
) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame()

    # Keep columns needed for merging and add source column
    in_df = sample_df[['chrom', 'start', 'end', 'source']].copy()

    # bedtools sort
    sort_cmd = ['bedtools', 'sort', '-g', genome_file_path, '-i', '-']

    # bedtools merge
    merge_cmd = ['bedtools', 'merge', '-i', '-', '-c', '4', '-o', 'distinct']

    with subprocess.Popen(
        sort_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as sort_p:
        with subprocess.Popen(
            merge_cmd,
            stdin=sort_p.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as merge_p:
            sort_p.stdout.close() if sort_p.stdout else None  # allow sort_p to get SIGPIPE if merge exits

            # Stream TSV directly into bedtools sort stdin
            in_df.to_csv(sort_p.stdin, sep='\t', index=False, header=False)
            sort_p.stdin.close() if sort_p.stdin else None  # signal end of input to sort

            merged_stdout, merged_stderr = merge_p.communicate()
            sort_stderr = sort_p.stderr.read() if sort_p.stderr else ""
            sort_rc = sort_p.wait()
    
    if sort_rc != 0:
        raise RuntimeError(f"bedtools sort failed for sample {sample_id} with error: {sort_stderr}")
    
    if merge_p.returncode != 0:
        raise RuntimeError(f"bedtools merge failed for sample {sample_id} with error: {merged_stderr}")
    
    if not merged_stdout.strip():
        return pd.DataFrame()  # No merged records

    # Parse merged output
    out_df = pd.read_csv(
        StringIO(merged_stdout), 
        sep='\t', 
        header=None,
        names=['chrom', 'start', 'end', 'source']
    )
    out_df['svtype'] = svtype
    out_df['sample_id'] = sample_id
    return out_df[['chrom', 'start', 'end', 'svtype', 'source', 'sample_id']]

def get_per_source_stats(sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        print("No benchmark records parsed before liftover.")
        return pd.DataFrame()

    source_record_counts = sample_df.groupby('source').size().rename('record_count')
    sample_counts_by_source = sample_df.groupby('source')['sample_id'].nunique().rename('sample_count')
    per_sample_counts = (
        sample_df.groupby(['source', 'sample_id'])
        .size()
        .rename('records_per_sample')
        .reset_index()
    )

    # Create temporary CNV size column for descriptive stats.
    sample_df_with_size = sample_df.copy()
    sample_df_with_size['size'] = sample_df_with_size['end'] - sample_df_with_size['start']

    size_stats = (
        sample_df_with_size.groupby('source')['size']
        .agg(['mean', 'median', 'min', 'max'])
        .rename(columns={'mean': 'size_mean', 'median': 'size_median', 'min': 'size_min', 'max': 'size_max'})
    )

    size_q1 = sample_df_with_size.groupby('source')['size'].quantile(0.25).rename('size_q1')
    size_q3 = sample_df_with_size.groupby('source')['size'].quantile(0.75).rename('size_q3')
    size_iqr = (size_q3 - size_q1).rename('size_iqr')

    del_counts = (
        sample_df_with_size[sample_df_with_size['svtype'] == 'DEL']
        .groupby('source')
        .size()
        .rename('del_count')
    )
    dup_counts = (
        sample_df_with_size[sample_df_with_size['svtype'] == 'DUP']
        .groupby('source')
        .size()
        .rename('dup_count')
    )

    pre_liftover_summary = (
        pd.concat(
            [
                source_record_counts,
                sample_counts_by_source,
                per_sample_counts.groupby('source')['records_per_sample'].mean().rename('records_per_sample_mean'),
                per_sample_counts.groupby('source')['records_per_sample'].median().rename('records_per_sample_median'),
                per_sample_counts.groupby('source')['records_per_sample'].min().rename('records_per_sample_min'),
                per_sample_counts.groupby('source')['records_per_sample'].max().rename('records_per_sample_max'),
                size_stats,
                size_q1,
                size_q3,
                size_iqr,
                del_counts,
                dup_counts,
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
    )

    # Additional derived QC metrics.
    pre_liftover_summary['avg_records_per_sample'] = (
        pre_liftover_summary['record_count'] / pre_liftover_summary['sample_count'].replace(0, pd.NA)
    )
    pre_liftover_summary['del_fraction'] = (
        pre_liftover_summary['del_count'] / pre_liftover_summary['record_count'].replace(0, pd.NA)
    )
    pre_liftover_summary['dup_fraction'] = (
        pre_liftover_summary['dup_count'] / pre_liftover_summary['record_count'].replace(0, pd.NA)
    )

    # Make numeric output easier to read in logs.
    numeric_cols = [
        'record_count', 'sample_count', 'records_per_sample_mean', 'records_per_sample_median',
        'records_per_sample_min', 'records_per_sample_max', 'avg_records_per_sample',
        'size_mean', 'size_median', 'size_min', 'size_max', 'size_q1', 'size_q3', 'size_iqr',
        'del_count', 'dup_count', 'del_fraction', 'dup_fraction'
    ]
    pre_liftover_summary[numeric_cols] = pre_liftover_summary[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    return pre_liftover_summary

def _parse_single_benchmark_from_path(
    vcf_path: str, 
    benchmark_name: str, 
    sample_ids: list, 
    valid_chroms: set
) -> List[Dict]:
    
    data: List[Dict] = []
    vcf = VCF(vcf_path, samples=sample_ids)
        
    for record in vcf:
        # Skip if chromosome is not in valid set
        chrom = ensure_chr_prefix(record.CHROM)
        if chrom not in valid_chroms:
            continue
            
        # Skip records without ALT alleles
        if not record.ALT or len(record.ALT) == 0:
            continue
            
        # Extract basic info
        start = record.POS - 1  # Convert to 0-based
        record_id = record.ID if record.ID else "."

        # Extract END - try INFO field first, then calculate from SVLEN
        end = record.INFO.get('END')
        if end is not None:
            end = int(end)
        else:
            svlen = record.INFO.get('SVLEN')
            if svlen is not None:
                end = record.POS + abs(int(svlen))
        
        # Skip if we couldn't determine END
        if end is None:
            continue
        
        # Extract and sanitize SVTYPE
        raw_svtype = record.INFO.get('SVTYPE')
        svtype = sanitize_svtype(raw_svtype, record_id)
        
        # Skip if SVTYPE is not DEL or DUP
        if svtype == 'NA':
            continue
        
        # Extract genotypes for requested samples
        genotypes = record.genotypes

        for idx, gt in enumerate(genotypes):
            if gt[0] == 0 and gt[1] == 0:
                continue  # Skip homozygous reference samples
            
            sample_id = vcf.samples[idx]
            data.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'svtype': svtype,
                'source': benchmark_name,
                'sample_id': sample_id
            })
    
    return data

def parse_benchmarks_to_bed(config: dict) -> tuple[pd.DataFrame, dict | None]:
    
    if 'benchmark_map' not in config:
        print("No benchmark map found in config. Skipping benchmark parsing.")
        return pd.DataFrame(), None

    layout = OutputLayout(Path(config['output_dir']))

    # Get common samples only
    sample_sets = []
    for _, vcf_path in config['benchmark_map'].items():
        try:
            vcf = VCF(vcf_path)
            sample_sets.append(set(vcf.samples))
        except Exception as e:
            print(f"Error reading samples from {vcf_path}: {e}")
            sample_sets.append(set())
    common_samples = set.intersection(*sample_sets) if sample_sets else set()
    print(f"Common samples across all benchmarks: {len(common_samples)}")
    sample_ids = list(common_samples)
    
    # Get valid chromosomes from genome file
    genome_file_path = config['genome_file']
    valid_chroms = set()
    try:
        with open(genome_file_path) as f:
            for line in f:
                chrom = line.split()[0]
                valid_chroms.add(ensure_chr_prefix(chrom))
    except Exception as e:
        print(f"Error reading genome file {genome_file_path}: {e}")
        print("Proceeding without chromosome filtering.")

    process_args_list = []
    for benchmark_name, vcf_path in config['benchmark_map'].items():
        process_args = (vcf_path, benchmark_name, sample_ids, valid_chroms)
        process_args_list.append(process_args)

    with ProcessPoolExecutor() as executor:
        results = executor.map(_parse_single_benchmark_from_path, *zip(*process_args_list))

    # Collect results from all benchmarks into a single list of records
    sample_data: List[Dict] = []
    for benchmark_name, records in zip(config['benchmark_map'].keys(), results):
        print(f"Parsed {len(records)} records from benchmark '{benchmark_name}'")
        sample_data.extend(records)

    # Covert to DataFrame
    sample_df = pd.DataFrame(sample_data)

    # Limit to chromosomes in config['valid_chromosomes'] if available
    if 'valid_chromosomes' in config:
        sample_df = sample_df[sample_df['chrom'].isin(config['valid_chromosomes'])]

    # pre_liftover_stats = get_per_source_stats(sample_df)
    # print("\nPre-liftover benchmark summary by source:")
    # print(pre_liftover_stats)

    # Perform liftover per source
    liftover_results = {}
    for source in sample_df['source'].unique():
        if source not in config['liftover']:
            continue

        source_df = sample_df[sample_df['source'] == source]
        from_build = config['liftover'][source]['from'] if config['liftover'].get(source) else None
        to_build = config['liftover'][source]['to'] if config['liftover'].get(source) else None

        if from_build and to_build:
            print(f"Performing liftover for benchmark '{source}' from {from_build} to {to_build}...")
            lifted_df, stats = perform_liftover(source_df, from_build, to_build)
            sample_df.loc[sample_df['source'] == source, ['chrom', 'start', 'end']] = lifted_df[['chrom', 'start', 'end']]
            liftover_results[source] = stats
            print(f"  Liftover completed for benchmark '{source}'.")

    # post_liftover_stats = get_per_source_stats(sample_df)
    # print("\nPost-liftover benchmark summary by source:")
    # print(post_liftover_stats)
    
    # Split across samples and merge nearby/overlapping records within each sample
    items = []
    output_dir = layout.benchmark
    os.makedirs(output_dir, exist_ok=True)
    for (sample_id, svtype), sample_df in sample_df.groupby(['sample_id', 'svtype']):
        items.append((
            sample_id, 
            svtype, 
            sample_df, 
            config['genome_file']
        ))

    # Setup and run parallel processing with ProcessPoolExecutor
    cpu_count = os.cpu_count()
    target_workers = max(1, (2 * cpu_count) // 3) if cpu_count else 1
    max_workers = min(len(items), target_workers) if items else 1
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_merge_one_sample, *zip(*items))
    
    # Collect results into a single DataFrame
    merged_sample_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    # tidy up source column by lowercasing, converting to set, and sorting values
    merged_sample_df['source'] = merged_sample_df['source'].apply(
        lambda x: ','.join(sorted(set(s.strip().lower() for s in x.split(','))))
    )

    # Perform liftover on merged records if configured for merged benchmarks
    if 'merged' in config['liftover']:
        from_build = config['liftover']['merged']['from']
        to_build = config['liftover']['merged']['to']

        print(f"Performing liftover for merged benchmarks from {from_build} to {to_build}...")
        
        lifted_df, stats = perform_liftover(merged_sample_df, from_build, to_build)
        liftover_results['merged'] = stats
        merged_sample_df = lifted_df
    
    # Write DEL and DUP files for each sample
    print("Exporting merged benchmarks to BED files...")

    output_dir = layout.benchmark
    os.makedirs(output_dir, exist_ok=True)
    for (sample_id, svtype), group_df in merged_sample_df.groupby(['sample_id', 'svtype']):
        sample_id_str = str(sample_id)

        if svtype == 'DEL':
            del_df = group_df[['chrom', 'start', 'end', 'svtype', 'source']]
            del_output = output_dir / f"{sample_id_str}.DEL.bed"
            del_df.to_csv(del_output, sep='\t', index=False, header=False)
        
        elif svtype == 'DUP':
            dup_df = group_df[['chrom', 'start', 'end', 'svtype', 'source']]
            dup_output = output_dir / f"{sample_id_str}.DUP.bed"
            dup_df.to_csv(dup_output, sep='\t', index=False, header=False)

    return merged_sample_df, liftover_results
