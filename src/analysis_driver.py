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


def load_sample_data(sample_id: str, sample_files: Dict[str, Dict[str, List[Path]]]) -> Dict:
    """
    Load all TP/FP/FN files for a single sample into dataframes.
    
    Args:
        sample_id: The sample identifier
        sample_files: Dictionary mapping svtype -> {classification -> [filepath]}
    
    Returns:
        Dictionary with 'dfs' and 'statistics' keys
    """
    sample_data = {
        'dfs': {},
        'statistics': {}
    }
    
    # Load each svtype (DEL, DUP)
    for svtype, classifications in sample_files.items():
        if svtype not in sample_data['dfs']:
            sample_data['dfs'][svtype] = {}
            sample_data['statistics'][svtype] = {}
        
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
            
            # Store dataframe and statistics
            sample_data['dfs'][svtype][classification] = df
            sample_data['statistics'][svtype][classification] = len(df)
    
    return sample_data


def build_analysis_data_structure(config: dict, input_set_key: str) -> Dict[str, Dict]:
    """
    Build the complete data structure for analysis.
    
    Args:
        config: The configuration dictionary from YAML
        input_set_key: The key for the input set (e.g., "Low Coverage", "High Coverage")
    
    Returns:
        Dictionary mapping sample_id -> {dfs: {...}, statistics: {...}}
    """
    output_dir = Path(config['output_dir'])
    
    # Convert input set key to directory name
    output_subdir_name = input_set_key.replace(" ", "_")
    binary_classification_dir = output_dir / output_subdir_name / "binary_classification"
    
    print(f"\nLoading data from: {binary_classification_dir}")
    
    # Discover all files
    files_by_sample = discover_classification_files(binary_classification_dir)
    
    # Build data structure
    analysis_data = {}
    for sample_id, sample_files in files_by_sample.items():
        print(f"Loading data for sample: {sample_id}")
        analysis_data[sample_id] = load_sample_data(sample_id, sample_files)
    
    return analysis_data


def print_summary_statistics(analysis_data: Dict[str, Dict]):
    """
    Print summary statistics for loaded data.
    
    Args:
        analysis_data: The complete analysis data structure
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for sample_id, sample_data in sorted(analysis_data.items()):
        print(f"\nSample: {sample_id}")
        print("-" * 40)
        
        for svtype in ['DEL', 'DUP']:
            if svtype in sample_data['statistics']:
                stats = sample_data['statistics'][svtype]
                tp_count = stats.get('TP', 0)
                fp_count = stats.get('FP', 0)
                fn_count = stats.get('FN', 0)
                
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
    input_sets = list(config['input'].keys())
    print(f"Available input sets: {input_sets}")
    
    # For now, load data from all input sets
    all_data = {}
    for input_set_key in input_sets:
        print(f"\n{'='*80}")
        print(f"Processing input set: {input_set_key}")
        print(f"{'='*80}")
        
        analysis_data = build_analysis_data_structure(config, input_set_key)
        all_data[input_set_key] = analysis_data
        
        # Print summary statistics
        print_summary_statistics(analysis_data)
    
    # TODO: Generate tables and plots from the loaded data
    # - Create summary tables (precision, recall, F1 scores)
    # - Generate plots (ROC curves, precision-recall curves, etc.)
    # - Export results to files
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()