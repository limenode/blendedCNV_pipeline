from pathlib import Path
import yaml
import pandas as pd
from typing import List, Tuple, Callable, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from load_analysis_data import build_analysis_data_structure, print_summary_statistics, filter_by_size
from plotting_functions import plot_performance_distributions, plot_venn_diagram_wrapper, plot_size_distribution

# Test Functions:
def _test1(all_data):
    from utils import precision, recall, f1_score

    # Get precision of high coverage input set for CNVs between 1kb and 2kb
    high_cov_data = all_data.get("High Coverage", {})
    if high_cov_data:
        filtered = filter_by_size(high_cov_data, lower_bound=1000, upper_bound=1500)
        tp_count = len(filtered.get('TP', pd.DataFrame()))
        fp_count = len(filtered.get('FP', pd.DataFrame()))
        fn_count = len(filtered.get('FN', pd.DataFrame()))
        precision_value = precision(tp_count, fp_count, fn_count)
        print(f"\nPrecision for High Coverage (1kb-2kb): {precision_value:.4f}")
    

def _load_data_for_all_input_sets(input_sets_paths: Dict[str, Path], shared_samples_only: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    all_data = {}
    sample_sets = []
    shared_samples = None

    # First pass to discover shared samples across classifications if needed
    for input_set_key, input_set_path in input_sets_paths.items():
        print(f"\n{'='*80}")
        print(f"Processing input set: {input_set_key}")
        print(f"{'='*80}")

        # Discover shared samples across TP, FP, FN for this input set
        if shared_samples_only:
            sample_sets = []
            # Use glob to discover all classification .bed files, then extract sample names
            for classification in ['TP', 'FP', 'FN']:
                if not input_set_path.exists():
                    print(f"Warning: Path '{input_set_path}' does not exist. Skipping sample discovery for '{classification}'.")
                    continue
                
                bed_files = list(input_set_path.glob("*.bed"))
                if not bed_files:
                    print(f"Warning: No .bed files found in '{input_set_path}'. Skipping.")
                    continue
                
                samples_in_classification = set()
                for bed_file in bed_files:
                    # Extract sample name from filename (assuming format: sample.<svtype>.<classification>.bed)
                    sample_name = bed_file.stem.split('.')[0]
                    samples_in_classification.add(sample_name)
                
                sample_sets.append(samples_in_classification)
    
    # Print information
    print("\nData loading complete for all input sets.")
    if shared_samples_only and shared_samples is not None:
        print(f"{len(shared_samples)} shared samples across classifications: {shared_samples}")
        
    # If we have sample sets, find the intersection (shared samples)
    if sample_sets:
        shared_samples = set.intersection(*sample_sets) if len(sample_sets) > 1 else sample_sets[0]
    else:
        shared_samples = None

    # Second pass to load data with optional filtering by shared samples
    for input_set_key, input_set_path in input_sets_paths.items():
        analysis_data = build_analysis_data_structure(input_set_path, samples_to_include=shared_samples)
        filtered_data = filter_by_size(analysis_data, lower_bound=500, upper_bound=1_000_000, strict=True)
        all_data[input_set_key] = filtered_data
    
    return all_data



def main(config: dict):
    """
    Main analysis pipeline.
    
    Args:
        config: Configuration dictionary loaded from YAML
    """

    # === Step 1: Prepare Input Set Paths ===

    # Get all input set keys
    input_sets_raw = list(config['input'].keys())
    print(f"Available input sets: {input_sets_raw}")

    # Append "Intersection" and "Union" to input set keys for binary classification results
    output_dir = Path(config['output_dir'])
    input_sets_paths = {}
    for key in input_sets_raw:
        key_path = key.replace(" ", "_")
        input_sets_paths[key_path + "_intersections"] = output_dir / key_path / "binary_classification" / "intersections"
        input_sets_paths[key_path + "_unions"] = output_dir / key_path / "binary_classification" / "unions"
    
    # Append control set
    input_sets_paths["SNP_Array"] = output_dir / "SNP_Array" / "binary_classification"
    


    # === Step 2: Load Data for All Input Sets ===
    all_data = _load_data_for_all_input_sets(input_sets_paths)

    print("\nSummary of loaded data:")
    for input_set_key, analysis_data in all_data.items():
        print("\ninput_set_key:", input_set_key)
        print("Keys in analysis_data:", analysis_data.keys())
        for classification_key, df in analysis_data.items():
            print(f"  Classification: {classification_key}, Number of records: {len(df)}")
    


    # # === Step 3: Generate Plots for All Distributions ===
    plot_performance_distributions(config, all_data)

    # # === Step 4: Generate Venn Diagrams ===
    plot_venn_diagram_wrapper(config, all_data)

    # === Step 5: Generate Size Distribution Plots ===
    plot_size_distribution(
        all_data=all_data,
        input_set_keys=list(all_data.keys()),
        svtype=None,
        output_path=output_dir / "size_distributions" / "size_distribution.png",
        title="Size Distribution of Detected CNVs"
    )
    

if __name__ == "__main__":
    # Allow running standalone for testing
    import argparse
    
    parser = argparse.ArgumentParser(description='BlendedCNV Analysis Pipeline')
    parser.add_argument('--config', '-c', required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

