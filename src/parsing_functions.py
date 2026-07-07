import os
from cyvcf2 import VCF
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor
import subprocess
from io import StringIO
import pandas as pd

from liftover import get_lifter
from utils import ensure_chr_prefix, sanitize_svtype, PipelineConfig

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

def parse_benchmarks_to_bed(config: PipelineConfig) -> tuple[pd.DataFrame, dict | None]:

    if not config.benchmark_map:
        print("No benchmark map found in config. Skipping benchmark parsing.")
        return pd.DataFrame(), None

    layout = config.layout

    # Get common samples only
    sample_sets = []
    for _, vcf_path in config.benchmark_map.items():
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
    genome_file_path = config.genome_file
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
    for benchmark_name, vcf_path in config.benchmark_map.items():
        process_args = (vcf_path, benchmark_name, sample_ids, valid_chroms)
        process_args_list.append(process_args)

    with ProcessPoolExecutor() as executor:
        results = executor.map(_parse_single_benchmark_from_path, *zip(*process_args_list))

    # Collect results from all benchmarks into a single list of records
    sample_data: List[Dict] = []
    for benchmark_name, records in zip(config.benchmark_map.keys(), results):
        print(f"Parsed {len(records)} records from benchmark '{benchmark_name}'")
        sample_data.extend(records)

    # Covert to DataFrame
    sample_df = pd.DataFrame(sample_data)

    # Limit to valid chromosomes if available
    if config.valid_chromosomes:
        sample_df = sample_df[sample_df['chrom'].isin(config.valid_chromosomes)]

    # Perform liftover per source
    liftover_results = {}
    for source in sample_df['source'].unique():
        if source not in config.liftover:
            continue

        source_df = sample_df[sample_df['source'] == source]
        from_build = config.liftover[source]['from'] if config.liftover.get(source) else None
        to_build = config.liftover[source]['to'] if config.liftover.get(source) else None

        if from_build and to_build:
            print(f"Performing liftover for benchmark '{source}' from {from_build} to {to_build}...")
            lifted_df, stats = perform_liftover(source_df, from_build, to_build)
            sample_df.loc[sample_df['source'] == source, ['chrom', 'start', 'end']] = lifted_df[['chrom', 'start', 'end']]
            liftover_results[source] = stats
            print(f"  Liftover completed for benchmark '{source}'.")

    # Split across samples and merge nearby/overlapping records within each sample
    items = []
    output_dir = layout.benchmark
    os.makedirs(output_dir, exist_ok=True)
    for (sample_id, svtype), sample_df in sample_df.groupby(['sample_id', 'svtype']):
        items.append((
            sample_id,
            svtype,
            sample_df,
            config.genome_file
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
    if 'merged' in config.liftover:
        from_build = config.liftover['merged']['from']
        to_build = config.liftover['merged']['to']

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
