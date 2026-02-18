from pathlib import Path
import yaml
import pandas as pd
from typing import List, Tuple, Callable, Dict, Optional
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from collections import defaultdict

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

def get_samples_from_data(all_data: Dict[str, Dict[str, pd.DataFrame]], classification_key: str) -> set:
    """Extract sample names from a specific classification across all input sets."""

    all_samples = set()
    for _, analysis_data in all_data.items():
        if classification_key in analysis_data:
            df = analysis_data[classification_key]
            if 'sample' in df.columns:
                all_samples.update(df['sample'].unique())
    return all_samples


def _compute_change_rows(consensus_calls_dict: Dict[str, list]) -> pd.DataFrame:
    rows = []
    for input_set, records in consensus_calls_dict.items():
        for rec in records:
            sample = rec.get("sample")
            svtype = rec.get("svtype")
            before = rec.get("before_excluded_regions", {})
            after = rec.get("after_excluded_regions", {})

            for caller in before.keys():
                before_val = before.get(caller, None)
                after_val = after.get(caller, None)
                if before_val is None or after_val is None:
                    continue

                abs_change = before_val - after_val  # reduction due to filtering
                pct_change = (abs_change / before_val * 100) if before_val > 0 else None

                rows.append({
                    "input_set": input_set,
                    "sample": sample,
                    "svtype": svtype,
                    "caller": caller,
                    "before": before_val,
                    "after": after_val,
                    "abs_change": abs_change,
                    "pct_change": pct_change,
                })
    return pd.DataFrame(rows)

def _plot_grouped_violin_plot(df: pd.DataFrame, value_col: str, title: str, output_path: Path, 
                          xlabel: str = "Caller", ylabel: Optional[str] = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get unique input sets and callers
    input_sets = sorted(df["input_set"].dropna().unique().tolist())
    callers = sorted(df["caller"].dropna().unique().tolist())
    
    num_input_sets = len(input_sets)
    num_callers = len(callers)
    
    # Create side-by-side subplots, one for each input set
    fig, axes = plt.subplots(1, num_input_sets, figsize=(6 * num_input_sets, 6), sharey=True)
    
    # Handle case where there's only one input set
    if num_input_sets == 1:
        axes = [axes]
    
    # Color palette for callers
    colors = matplotlib.colormaps['Set3'](np.linspace(0, 1, num_callers))
    
    for idx, (input_set, ax) in enumerate(zip(input_sets, axes)):
        # Plot violins for each caller in this input set
        for j, caller in enumerate(callers):
            subset = df[(df["input_set"] == input_set) & (df["caller"] == caller)][value_col].dropna()
            data = subset.values
            
            if len(data) > 0:
                # Create violin plot
                parts = ax.violinplot([data], positions=[j + 1], widths=0.6, 
                                     showmeans=False, showmedians=True)
                
                # Color violin body
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[j])
                    pc.set_alpha(0.3)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1)
                
                # Style the other violin components
                for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                    if partname in parts:
                        vp = parts[partname]
                        vp.set_edgecolor('black')
                        vp.set_linewidth(1)
                
                # Overlay individual points with jitter
                jitter = np.random.normal(loc=0, scale=0.08, size=len(data))
                ax.scatter(np.full(len(data), j + 1) + jitter, data, s=12, 
                          alpha=0.6, color='black', zorder=3)
        
        # Set subplot title and labels
        ax.set_title(input_set, fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, num_callers + 1))
        ax.set_xticklabels(callers)
        ax.set_xlabel(xlabel)
        
        # Only set y-label on the leftmost subplot
        if idx == 0:
            ax.set_ylabel(ylabel if ylabel else value_col.replace("_", " ").title())
    
    # Set overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def load_logs(log_dir: Path, samples: Optional[List[str]] = None):
    # benchmark_processing_results.json
    benchmark_merging_file = log_dir / "benchmark_processing_results.json"
    benchmark_merging_df = pd.read_json(benchmark_merging_file).T
    print("\nBenchmark Merging Results:")
    print(benchmark_merging_df.head())

    # consensus_calls_results.json
    consensus_calls_file = log_dir / "consensus_calls_results.json"
    consensus_calls_dict = json.loads(consensus_calls_file.read_text())

    changes_df = _compute_change_rows(consensus_calls_dict)
    
    # Filter by samples if provided
    if samples is not None:
        sample_set = set(samples)
        original_count = len(changes_df)
        changes_df = changes_df[changes_df["sample"].isin(sample_set)].copy()
        filtered_count = len(changes_df)
        print(f"\nFiltered to {len(sample_set)} specified samples ({original_count - filtered_count} rows removed)")
    
    print("\nConsensus Calls Change Summary:")
    print(changes_df.head())

    # Group statistics by input_set and caller
    grouped_means = (
        changes_df.groupby(["input_set", "caller"], dropna=True)[["abs_change", "pct_change"]]
        .mean()
        .reset_index()
        .sort_values(["input_set", "caller"])
    )
    print("\nMean Changes by Input Set and Caller:")
    print(grouped_means)

    # Count data points per input_set and caller
    abs_counts = changes_df.groupby(["input_set", "caller"], dropna=True)["abs_change"].count()
    pct_counts = changes_df.groupby(["input_set", "caller"], dropna=True)["pct_change"].count()
    print("\nBox Plot Data Point Counts (abs_change):")
    print(abs_counts)
    print("\nBox Plot Data Point Counts (pct_change):")
    print(pct_counts)

    figures_dir = log_dir / "figures"
    _plot_grouped_violin_plot(
        changes_df,
        value_col="abs_change",
        title="Absolute Change After Excluding Regions",
        output_path=figures_dir / "abs_change_boxplot.png",
        xlabel="Caller",
        ylabel="Absolute Change (# of calls removed)",
    )
    _plot_grouped_violin_plot(
        changes_df,
        value_col="pct_change",
        title="Percent Change After Excluding Regions",
        output_path=figures_dir / "pct_change_boxplot.png",
        xlabel="Caller",
        ylabel="Percent Change (%)",
    )
    
def get_caller_source_distribution(
        all_data: Dict[str, Dict[str, pd.DataFrame]], 
        input_sets_to_include: List[str], 
        output_dir: Path
    ):
    """
    Analyze caller source distributions per sample and svtype, then generate box plots.
    
    Args:
        all_data: Dictionary of analysis data per input set
        output_dir: Directory to save plots
    """
    rows = []

    for input_set_key, analysis_data in all_data.items():
        if input_set_key not in input_sets_to_include:
            print(f"Skipping input set '{input_set_key}' for caller source distribution analysis")
            continue

        if "TP" in analysis_data:
            tp_df = analysis_data["TP"]
            if "sources" in tp_df.columns and "sample" in tp_df.columns and "svtype" in tp_df.columns:
                # Group by sample and svtype
                for (sample, svtype), group in tp_df.groupby(["sample", "svtype"]):
                    raw_caller_counts = defaultdict(int)
                    combination_counts = defaultdict(int)
                    
                    total_calls = len(group["sources"].dropna())
                    
                    if total_calls == 0:
                        continue
                    
                    for source_list in group["sources"].dropna():
                        sources = source_list.split("|")
                        
                        # Count raw caller occurrences
                        for source in sources:
                            raw_caller_counts[source] += 1
                        
                        # Count combinations
                        combination_key = "|".join(sorted(sources))
                        combination_counts[combination_key] += 1
                    
                    # Add raw caller percentages as separate rows
                    for caller, count in raw_caller_counts.items():
                        percentage = (count / total_calls) * 100
                        rows.append({
                            "input_set": input_set_key,
                            "sample": sample,
                            "svtype": svtype,
                            "metric": "raw_count",
                            "caller_or_combination": caller,
                            "percentage": percentage,
                        })
                    
                    # Add combination percentages as separate rows
                    for combination, count in combination_counts.items():
                        percentage = (count / total_calls) * 100
                        rows.append({
                            "input_set": input_set_key,
                            "sample": sample,
                            "svtype": svtype,
                            "metric": "combination_count",
                            "caller_or_combination": combination,
                            "percentage": percentage,
                        })

    df = pd.DataFrame(rows)
    
    if df.empty:
        print("No source distribution data found")
        return df
    
    print("\nCaller Source Distribution Summary:")
    print(df.head())
    
    figures_dir = output_dir / "caller_source_distributions"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 4 plots: DEL raw, DEL combo, DUP raw, DUP combo
    for svtype in ["DEL", "DUP"]:
        for metric in ["raw_count", "combination_count"]:
            subset = df[(df["svtype"] == svtype) & (df["metric"] == metric)]
            
            if subset.empty:
                print(f"No data for {svtype} {metric}")
                continue
            
            # Get unique callers/combinations
            entities = sorted(subset["caller_or_combination"].unique())
            
            fig, ax = plt.subplots(figsize=(max(10, len(entities) * 1.5), 6))
            
            # Prepare data for box plot
            data_to_plot = []
            labels = []
            for entity in entities:
                entity_data = subset[subset["caller_or_combination"] == entity]["percentage"].values
                if len(entity_data) > 0:
                    data_to_plot.append(entity_data)
                    labels.append(entity)
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showfliers=True)
                
                # Style boxes
                for box in bp['boxes']:
                    box.set_facecolor('lightblue')
                    box.set_alpha(0.5)
                
                # Overlay individual points with jitter
                for i, vals in enumerate(data_to_plot, start=1):
                    jitter = np.random.normal(loc=0, scale=0.04, size=len(vals))
                    ax.scatter(np.full(len(vals), i) + jitter, vals, s=15, alpha=0.6, color='black')
            
            metric_label = "Raw Caller" if metric == "raw_count" else "Caller Combination"
            ax.set_title(f"{svtype} {metric_label} Distribution", fontsize=14, fontweight='bold')
            ax.set_ylabel("Percentage per Sample (%)")
            ax.set_xlabel("Caller" if metric == "raw_count" else "Caller Combination")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            filename = f"{svtype.lower()}_{metric}_boxplot.png"
            plt.savefig(figures_dir / filename, dpi=150)
            plt.close()
            print(f"Saved {filename}")
    
    return df

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

    # === Test Step 1: Load logs ===
    log_dir = Path(config['output_dir']) / "logs"
    samples_of_interest = ['HG01890', 'NA19347', 'HG00513', 'HG01596', 'NA19238', 'NA19331', 'HG00096', 'HG00171', 'NA18989', 'HG00268', 'NA20847', 'HG00731', 'NA19129']
    load_logs(log_dir, samples=samples_of_interest)


    # === Step 2: Load Data for All Input Sets ===
    all_data = _load_data_for_all_input_sets(input_sets_paths)
    samples = get_samples_from_data(all_data, classification_key='TP')

    sets_to_include_for_distribution = [key for key in all_data.keys() if "intersections" in key]
    get_caller_source_distribution(all_data, sets_to_include_for_distribution, output_dir)

    return

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

    # === Step 4: Generate Venn Diagrams ===
    plotter.plot_venn_diagram(
        set_keys=['Low_Coverage_intersections', 'High_Coverage_intersections', 'SNP_Array'],
        output_path=output_dir / "venn_diagrams" / "venn_diagram_intersections.png",
    )
    plotter.plot_venn_diagram(
        set_keys=['Low_Coverage_unions', 'High_Coverage_unions', 'SNP_Array'],
        output_path=output_dir / "venn_diagrams" / "venn_diagram_unions.png",
    )

    # === Step 5: Generate Size Distribution Plots ===
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

