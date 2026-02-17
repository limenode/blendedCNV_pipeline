from pathlib import Path
import yaml
import pandas as pd
from typing import List, Tuple, Callable, Dict, Optional

from load_analysis_data import build_analysis_data_structure, print_summary_statistics, filter_by_size
from cnv_plotter import CNVPlotter
from utils import precision, recall, f1_score
    

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

    # Setup input name mapping for user-friendly display
    input_name_mapping = {}

    # Append "Intersection" and "Union" to input set keys for binary classification results
    output_dir = Path(config['output_dir'])
    input_sets_paths = {}
    for key in input_sets_raw:
        key_path = key.replace(" ", "_")
        input_sets_paths[key_path + "_intersections"] = output_dir / key_path / "binary_classification" / "intersections"
        input_sets_paths[key_path + "_unions"] = output_dir / key_path / "binary_classification" / "unions"

        # Add to input name mapping
        input_name_mapping[key_path + "_intersections"] = f"{key} Intersections"
        input_name_mapping[key_path + "_unions"] = f"{key} Unions"
    
    # Append control sets
    control_sets_raw = list(config['control'].keys())
    for key in control_sets_raw:
        key_path = key.replace(" ", "_")
        input_sets_paths[key_path] = output_dir / key_path / "binary_classification"
        input_name_mapping[key_path] = key    


    # === Step 2: Load Data for All Input Sets ===
    all_data = _load_data_for_all_input_sets(input_sets_paths)

    print("\nSummary of loaded data:")
    for input_set_key, analysis_data in all_data.items():
        print("\ninput_set_key:", input_set_key)
        print("Keys in analysis_data:", analysis_data.keys())
        for classification_key, df in analysis_data.items():
            print(f"  Classification: {classification_key}, Number of records: {len(df)}")
    
    plotter = CNVPlotter(all_data, config, input_name_mapping)

    metrics = [(precision, "Precision"), (recall, "Recall"), (f1_score, "F1 Score")]

    # === Step 3: Generate Plots for All Distributions ===
    plotter.plot_statistical_distributions(
        metrics=metrics,
        bounds=(500, 1_000_000),
        output_path=output_dir / "statistical_distributions" / "distribution.png",
    )

    # # === Step 4: Generate Venn Diagrams ===
    plotter.plot_venn_diagram(
        set_keys=['Low_Coverage_intersections', 'High_Coverage_intersections', 'SNP_Array'],
        output_path=output_dir / "venn_diagrams" / "venn_diagram_intersections.png",
    )
    plotter.plot_venn_diagram(
        set_keys=['Low_Coverage_unions', 'High_Coverage_unions', 'SNP_Array'],
        output_path=output_dir / "venn_diagrams" / "venn_diagram_unions.png",
    )

    # # === Step 5: Generate Size Distribution Plots ===
    plotter.plot_size_distribution(
        set_keys=list(all_data.keys()),
        output_dir=output_dir / "size_distributions",
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

