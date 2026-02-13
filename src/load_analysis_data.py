from utils import parse_args
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


pred_columns = ['pred_chrom', 'pred_start', 'pred_end', 'svtype', 'sources']
truth_columns = ['truth_chrom', 'truth_start', 'truth_end']


def load_tp_file(filepath: Path) -> pd.DataFrame:
    """
    Load a True Positive (TP) BED file into a DataFrame.
    
    TP files contain 8 columns:
    - predicted_record: chrom, start, end, svtype, sources (5 columns)
    - truth_record: chrom, start, end (3 columns)
    """
    columns = pred_columns + truth_columns
    
    df = pd.read_csv(filepath, sep='\t', header=None, names=columns)

    # Add size columns for predicted and truth records
    df['pred_size'] = df['pred_end'] - df['pred_start']
    df['truth_size'] = df['truth_end'] - df['truth_start']

    return df


def load_fp_file(filepath: Path) -> pd.DataFrame:
    """
    Load a False Positive (FP) BED file into a DataFrame.
    
    FP files contain 5 columns (predicted record only):
    - chrom, start, end, svtype, sources
    """    
    df = pd.read_csv(filepath, sep='\t', header=None, names=pred_columns)

    # Add size column for predicted records
    df['pred_size'] = df['pred_end'] - df['pred_start']

    return df


def load_fn_file(filepath: Path) -> pd.DataFrame:
    """
    Load a False Negative (FN) BED file into a DataFrame.
    
    FN files contain 3 columns (truth record only):
    - chrom, start, end
    """
    
    df = pd.read_csv(filepath, sep='\t', header=None, names=truth_columns)

    # Add size column for truth records
    df['truth_size'] = df['truth_end'] - df['truth_start']

    return df


def discover_classification_files(binary_classification_dir: Path) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    """
    Discover all TP/FP/FN files in the binary classification directory.
    
    Files follow the pattern: {sample_id}.{svtype}.{classification}.bed
    
    Returns:
        Dictionary mapping sample_id -> {svtype -> {classification -> [filepath]}}
    """
    files_by_sample = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    if not binary_classification_dir.exists():
        print(f"Warning: Directory {binary_classification_dir} does not exist")
        return dict(files_by_sample) # type: ignore
    
    # Find all .bed files
    for bed_file in binary_classification_dir.glob("*.bed"):
        # Parse filename: {sample_id}.{svtype}.{classification}.bed
        parts = bed_file.stem.split('.')
        if len(parts) == 3:
            sample_id, svtype, classification = parts
            files_by_sample[sample_id][svtype][classification].append(bed_file)
    
    return dict(files_by_sample) # type: ignore


def load_sample_data(sample_id: str, sample_files: Dict[str, Dict[str, List[Path]]]) -> Dict[str, pd.DataFrame]:
    """
    Load all TP/FP/FN files for a single sample into dataframes with metadata.
    
    Args:
        sample_id: The sample identifier
        sample_files: Dictionary mapping svtype -> {classification -> [filepath]}
    
    Returns:
        Dictionary mapping classification -> dataframe (with 'sample' and 'svtype' columns added)
    """
    classification_dfs = defaultdict(list)
    
    # Load each svtype (DEL, DUP)
    for svtype, classifications in sample_files.items():
        # Load each classification (TP, FP, FN)
        for classification, filepaths in classifications.items():
            if len(filepaths) == 0:
                continue
            
            # Use the first file (should only be one per combination)
            filepath = filepaths[0]
            
            # Load based on classification type
            if classification == 'TP':
                df = load_tp_file(filepath)
            elif classification == 'FP':
                df = load_fp_file(filepath)
            elif classification == 'FN':
                df = load_fn_file(filepath)
            else:
                print(f"Warning: Unknown classification '{classification}' for {filepath}")
                continue
            
            # Add metadata columns
            df['sample'] = sample_id
            df['svtype'] = svtype  # Override for TP/FP, add for FN
            
            # Store for concatenation
            classification_dfs[classification].append(df)
    
    # Concatenate dataframes for each classification
    result = {}
    for classification, dfs_list in classification_dfs.items():
        if dfs_list:
            result[classification] = pd.concat(dfs_list, ignore_index=True)
    
    return result


def build_analysis_data_structure(binary_classification_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Build the complete data structure for analysis.
    
    Args:
        binary_classification_dir: Path to the directory containing TP/FP/FN files for the input set.
    
    Returns:
        Dictionary mapping classification ('TP', 'FP', 'FN') -> concatenated dataframe
        Each dataframe contains all samples with 'sample' and 'svtype' columns
    """
    
    print(f"\nLoading data from: {binary_classification_dir}")
    
    # Discover all files
    files_by_sample = discover_classification_files(binary_classification_dir)
    
    # Collect dataframes by classification across all samples
    classification_dfs = defaultdict(list)
    
    for sample_id, sample_files in files_by_sample.items():
        print(f"Loading data for sample: {sample_id}")
        sample_data = load_sample_data(sample_id, sample_files)
        
        # Add to classification collections
        for classification, df in sample_data.items():
            classification_dfs[classification].append(df)
    
    # Concatenate all dataframes for each classification
    analysis_data = {}
    for classification in ['TP', 'FP', 'FN']:
        if classification in classification_dfs and classification_dfs[classification]:
            analysis_data[classification] = pd.concat(
                classification_dfs[classification], 
                ignore_index=True
            )
            # print(f"  Total {classification} records: {len(analysis_data[classification])}")
        else:
            # Create empty dataframe with appropriate columns
            if classification == 'TP':
                analysis_data[classification] = pd.DataFrame(
                    columns=pred_columns + truth_columns + ['pred_size', 'truth_size', 'sample', 'svtype']
                )
            elif classification == 'FP':
                analysis_data[classification] = pd.DataFrame(
                    columns=pred_columns + ['pred_size', 'sample', 'svtype']
                )
            elif classification == 'FN':
                analysis_data[classification] = pd.DataFrame(
                    columns=truth_columns + ['truth_size', 'sample', 'svtype']
                )
            print(f"  No {classification} records found")
    
    return analysis_data


def filter_by_size(analysis_data: Dict[str, pd.DataFrame], lower_bound: int, upper_bound: int) -> Dict[str, pd.DataFrame]:
    """
    Filter all dataframes in analysis_data by CNV size.
    
    For predicted sizes: strict filter [lower_bound, upper_bound]
    For truth sizes: relaxed filter [lower_bound * 0.50, upper_bound / 0.50]
    
    The relaxed filter for truth sizes accommodates the 50% reciprocal overlap
    requirement used in TP calculation. A truth CNV can be 0.5x to 2x the size
    of a predicted CNV and still achieve 50% reciprocal overlap.
    
    Args:
        analysis_data: Dictionary mapping classification -> dataframe
        lower_bound: Lower size bound (inclusive) for predicted CNVs
        upper_bound: Upper size bound (inclusive) for predicted CNVs
    
    Returns:
        Filtered analysis data with the same structure
    """
    # Calculate relaxed bounds for truth sizes
    truth_lower = lower_bound * 0.50
    truth_upper = upper_bound / 0.50
    
    filtered_data = {}
    
    for classification, df in analysis_data.items():
        if df.empty:
            filtered_data[classification] = df.copy()
            continue
        
        # Apply filters based on classification type
        if classification == 'TP':
            # TP has both pred_size and truth_size
            filtered_df = df[
                (df['pred_size'] >= lower_bound) & 
                (df['pred_size'] <= upper_bound) &
                (df['truth_size'] >= truth_lower) &
                (df['truth_size'] <= truth_upper)
            ].copy()
            
        elif classification == 'FP':
            # FP only has pred_size
            filtered_df = df[
                (df['pred_size'] >= lower_bound) & 
                (df['pred_size'] <= upper_bound)
            ].copy()
            
        elif classification == 'FN':
            # FN only has truth_size
            filtered_df = df[
                (df['truth_size'] >= truth_lower) &
                (df['truth_size'] <= truth_upper)
            ].copy()
        else:
            # Unknown classification, copy as-is
            filtered_df = df.copy()
        
        filtered_data[classification] = filtered_df
    
    return filtered_data


def print_summary_statistics(analysis_data: Dict[str, pd.DataFrame], title: Optional[str] = None):
    """
    Print summary statistics for loaded data.
    
    Args:
        analysis_data: Dictionary mapping classification -> dataframe
        title: Optional title to display (e.g., "Size Range: 1000-10000")
    """
    print("\n" + "="*80)
    if title:
        print(f"SUMMARY STATISTICS - {title}")
    else:
        print("SUMMARY STATISTICS")
    print("="*80)
    
    # Get unique samples and svtypes from the data
    samples = set()
    svtypes = set()
    
    for classification, df in analysis_data.items():
        if not df.empty:
            samples.update(df['sample'].unique())
            svtypes.update(df['svtype'].unique())
    
    samples = sorted(samples)
    svtypes = sorted(svtypes)
    
    # Print statistics for each sample and svtype combination
    for sample_id in samples:
        print(f"\nSample: {sample_id}")
        print("-" * 40)
        
        for svtype in svtypes:
            # Count records for this sample and svtype
            tp_count = len(analysis_data.get('TP', pd.DataFrame()).query(
                f"sample == '{sample_id}' and svtype == '{svtype}'"
            )) if 'TP' in analysis_data and not analysis_data['TP'].empty else 0
            
            fp_count = len(analysis_data.get('FP', pd.DataFrame()).query(
                f"sample == '{sample_id}' and svtype == '{svtype}'"
            )) if 'FP' in analysis_data and not analysis_data['FP'].empty else 0
            
            fn_count = len(analysis_data.get('FN', pd.DataFrame()).query(
                f"sample == '{sample_id}' and svtype == '{svtype}'"
            )) if 'FN' in analysis_data and not analysis_data['FN'].empty else 0
            
            # Calculate metrics
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  {svtype}:")
            print(f"    TP: {tp_count:4d}  |  FP: {fp_count:4d}  |  FN: {fn_count:4d}")
            print(f"    Precision: {precision:.3f}  |  Recall: {recall:.3f}  |  F1: {f1:.3f}")


def main():
    # Parse command-line arguments
    args = parse_args()
        
    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get all input set keys
    input_sets_raw = list(config['input'].keys())
    print(f"Available input sets: {input_sets_raw}")

    # Append "Intersection" and "Union" to input set keys for binary classification results
    output_dir = Path(config['output_dir'])
    input_sets_paths = []
    for key in input_sets_raw:
        key_path = key.replace(" ", "_")
        input_sets_paths.append(output_dir / key_path / "binary_classification" / "intersections")
        input_sets_paths.append(output_dir / key_path / "binary_classification" / "unions")

    
    # For now, load data from all input sets
    all_data = {}
    for input_set_path in input_sets_paths:
        print(f"\n{'='*80}")
        print(f"Processing input set at: {input_set_path}")
        print(f"{'='*80}")
        
        analysis_data = build_analysis_data_structure(input_set_path)
        all_data[input_set_path] = analysis_data
        
        # Print summary statistics for all data
        print_summary_statistics(analysis_data, title="All Sizes")
        
        # Example: Filter by size range
        # Common size ranges for CNV analysis:
        # - Small: 1kb - 10kb
        # - Medium: 10kb - 100kb
        # - Large: 100kb - 1Mb
        # - Very Large: > 1Mb
        
        # Example filter for medium-sized CNVs (10kb - 100kb)
        filtered_medium = filter_by_size(analysis_data, lower_bound=10000, upper_bound=100000)
        print_summary_statistics(filtered_medium, title="Size Range: 10kb - 100kb")
        
        # Example filter for large CNVs (100kb - 1Mb)
        filtered_large = filter_by_size(analysis_data, lower_bound=100000, upper_bound=1000000)
        print_summary_statistics(filtered_large, title="Size Range: 100kb - 1Mb")
        
        # Show easy data access examples
        print("\n" + "="*80)
        print("DATA ACCESS EXAMPLES")
        print("="*80)
        print(f"\nTotal TP records: {len(analysis_data.get('TP', pd.DataFrame()))}")
        print(f"Total FP records: {len(analysis_data.get('FP', pd.DataFrame()))}")
        print(f"Total FN records: {len(analysis_data.get('FN', pd.DataFrame()))}")
        
        if 'TP' in analysis_data and not analysis_data['TP'].empty:
            # Example: Get all DEL-type TPs
            del_tps = analysis_data['TP'][analysis_data['TP']['svtype'] == 'DEL']
            print(f"\nDEL-type TPs: {len(del_tps)}")
            
            # Example: Get TPs for a specific sample
            if len(analysis_data['TP']) > 0:
                sample_example = analysis_data['TP']['sample'].iloc[0]
                sample_tps = analysis_data['TP'][analysis_data['TP']['sample'] == sample_example]
                print(f"TPs for sample {sample_example}: {len(sample_tps)}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
