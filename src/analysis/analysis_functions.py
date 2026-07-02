from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict, Optional
import json
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analysis.load_analysis_data import build_analysis_data_structure, filter_by_size
from analysis.cnv_plotter import _create_record_ids
from utils import PipelineConfig
    

def load_data_for_all_input_sets(
        input_sets_paths: Dict[str, Path], 
        shared_samples_only: bool = True, 
        bounds: Tuple[int, int] = (500, 1_000_000)
    ) -> Dict[str, dict]:

    all_input_sets_data = {}
    
    # First pass: discover shared samples using filename parsing
    print(f"\n{'='*80}")
    print("Discovering samples across input sets...")
    print(f"{'='*80}")
    
    all_samples_per_input_set = {}
    for input_set_key, input_set_path in input_sets_paths.items():
        if not input_set_path.exists():
            print(f"Warning: Path '{input_set_path}' does not exist. Skipping {input_set_key}.")
            continue
        
        # Get sample names efficiently from filenames (text before first dot)
        bed_files = list(input_set_path.glob("*.bed"))
        samples_in_input_set = {bed_file.stem.split('.')[0] for bed_file in bed_files}
        
        all_samples_per_input_set[input_set_key] = samples_in_input_set
        print(f"  {input_set_key}: {len(samples_in_input_set)} samples")
    
    # Determine shared samples
    shared_samples = None
    if shared_samples_only and all_samples_per_input_set:
        shared_samples = set.intersection(*all_samples_per_input_set.values())
        print(f"\nShared samples across all input sets: {len(shared_samples)}")
    
    # Second pass: load data with optional filtering by shared samples
    print(f"\n{'='*80}")
    print("Loading and filtering data...")
    print(f"{'='*80}")
    
    for input_set_key, input_set_path in input_sets_paths.items():
        if not input_set_path.exists():
            continue
        
        print(f"  Processing: {input_set_key}")
        analysis_data = build_analysis_data_structure(input_set_path, samples_to_include=shared_samples)
        filtered_data = filter_by_size(analysis_data, lower_bound=bounds[0], upper_bound=bounds[1], strict=True)
        all_input_sets_data[input_set_key] = filtered_data
    
    # Third pass: Compute shared FNs efficiently using vectorized operations
    print(f"\n{'='*80}")
    print("Computing shared FNs (present in ALL samples across ALL input sets)...")
    print(f"{'='*80}")
    
    shared_fn_data = {'FN': pd.DataFrame()}
    
    if len(all_input_sets_data) > 0:
        # Collect all FN DataFrames with input set labels
        all_fn_dfs = []
        for input_set_key, analysis_data in all_input_sets_data.items():
            fn_df = analysis_data.get('FN', pd.DataFrame())
            if not fn_df.empty:
                fn_df_copy = fn_df.copy()
                fn_df_copy['_input_set'] = input_set_key
                all_fn_dfs.append(fn_df_copy)
                print(f"  {input_set_key}: {len(fn_df)} FNs")
        
        if all_fn_dfs:
            # Combine all FN data
            combined_fn_df = pd.concat(all_fn_dfs, ignore_index=True)
            
            # Create unique identifier for each FN record using vectorized operations
            combined_fn_df['_fn_id'] = (
                combined_fn_df['truth_chrom'].astype(str) + '_' +
                combined_fn_df['truth_start'].astype(str) + '_' +
                combined_fn_df['truth_end'].astype(str) + '_' +
                combined_fn_df['svtype'].astype(str)
            )
            
            # Count occurrences across input sets and samples
            fn_counts = combined_fn_df.groupby('_fn_id').agg({
                '_input_set': 'nunique',
                'sample': 'nunique'
            }).reset_index()
            
            num_input_sets = len(all_input_sets_data)
            num_samples = len(shared_samples) if shared_samples else combined_fn_df['sample'].nunique()
            
            # Find FNs present in all input sets AND all samples
            shared_fn_ids = fn_counts[
                (fn_counts['_input_set'] == num_input_sets) & 
                (fn_counts['sample'] == num_samples)
            ]['_fn_id'].tolist()
            
            print(f"\n  Total unique FNs: {fn_counts.shape[0]}")
            print(f"  FNs in all {num_input_sets} input sets and all {num_samples} samples: {len(shared_fn_ids)}")
            
            if shared_fn_ids:
                # Get representative records for shared FNs (one per FN)
                shared_fn_mask = combined_fn_df['_fn_id'].isin(shared_fn_ids)
                shared_fn_df = combined_fn_df[shared_fn_mask].drop_duplicates(subset='_fn_id', keep='first').copy()
                shared_fn_df = shared_fn_df.drop(columns=['_fn_id', '_input_set'])
                
                shared_fn_data = {'FN': shared_fn_df}
                print(f"  Stored {len(shared_fn_df)} shared FN records")
        else:
            print("  No FN data found in any input set")
    
    print(f"{'='*80}\n")
    
    # Return structured data with input_sets separated from shared_FN
    return {
        'input_sets': all_input_sets_data,
        'shared_FN': shared_fn_data
    }

def get_bed_counts(directory: Path, 
                   bounds: Optional[Tuple[int, int]] = None, 
                   script_path: Optional[Path] = None,
                   samples: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Call get_bed_counts.sh on a single directory and return sample counts.
    
    Args:
        directory: Path to directory containing .bed files
        bounds: Optional tuple (lower_bound, upper_bound) for CNV size filtering
        script_path: Optional path to get_bed_counts.sh (defaults to same dir as this file)
        samples: Optional list of sample names to include (filters files by sample name)
    
    Returns:
        Dictionary mapping sample_id to count: {'sample_1': count, 'sample_2': count, ...}
    """
    if script_path is None:
        script_path = Path(__file__).parent / 'get_bed_counts.sh'
    
    script_path = Path(script_path)
    directory = Path(directory)
    
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    lower_bound = bounds[0] if bounds else None
    upper_bound = bounds[1] if bounds else None

    # Build command
    cmd = [str(script_path), str(directory)]
    if lower_bound is not None:
        cmd.append(str(lower_bound))
        if upper_bound is not None:
            cmd.append(str(upper_bound))
    elif upper_bound is not None:
        # If only upper bound, need empty string for lower bound
        cmd.extend(['', str(upper_bound)])
    
    # Add samples parameter if provided
    if samples:
        # If bounds weren't specified, need to fill in empty strings
        while len(cmd) < 4:
            cmd.append('')
        cmd.append(','.join(samples))
    
    # Run script and capture output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )
    
    # Parse and return JSON output
    return json.loads(result.stdout)

def get_counts_from_config(config: PipelineConfig,
                           bounds: Optional[Tuple[int, int]] = None,
                           samples: Optional[List[str]] = None) -> Tuple[dict, dict]:
    """
    Run get_bed_counts.sh on multiple directories specified in config.
    
    Args:
        config: Dictionary containing:
            - 'input_sets': Dict mapping input_set_name to directory path
            - 'bounds': Optional tuple (lower, upper) for size filtering
            - 'script_path': Optional path to get_bed_counts.sh (defaults to same dir as this file)
        bounds: Optional tuple (lower, upper) for CNV size filtering
        samples: Optional list of sample names to include (filters files by sample name)
    
    Returns:
        Tuple of (raw_results, post_processed_results):
        - raw_results: Dictionary mapping input_set_name to sample counts
        - post_processed_results: Dictionary with aggregated counts by svtype 
    """
    results = {}

    layout = config.layout

    consensus_types = ['consensus_1of3', 'consensus_2of3', 'consensus_3of3']

    sets_to_process = {}
    input_names = []
    control_names = []

    for key, path in config.input.items():
        for consensus_type in consensus_types:
            output_subdir = layout.set_dir(key) / consensus_type
            sets_to_process[f"{key}.{consensus_type}"] = output_subdir
            input_names.append(f"{key}.{consensus_type}")

    for key, path in config.control.items():
        output_subdir = layout.control_bed_dir(key)
        sets_to_process[key] = output_subdir
        control_names.append(key)

    if config.benchmark_map:
        sets_to_process['Benchmark'] = layout.benchmark

    script_path = Path("./src/get_bed_counts.sh")

    # Process each input set
    for set_name, directory in sets_to_process.items():
        directory = Path(directory)
        
        if not directory.exists():
            print(f"Warning: Directory '{directory}' does not exist. Skipping {set_name}.")
            continue
        
        print(f"Getting counts for {set_name} from {directory}")
        
        try:
            counts_dict = get_bed_counts(directory, bounds, script_path, samples)
            results[set_name] = counts_dict
            
            print(f"  Found {len(counts_dict)} samples with total {sum(counts_dict.values())} CNVs")
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error processing {set_name}: {e}")
            results[set_name] = {}
    
    # Post-processing

    post_results = {}
    
    # Input sets - aggregate by intersections/unions and svtype
    for input_set_name in input_names:
        if input_set_name not in results or not results[input_set_name]:
            continue
        
        raw_counts = results[input_set_name]
        aggregated = {
            'intersections': {'DEL': 0, 'DUP': 0, 'ALL': 0},
            'unions': {'DEL': 0, 'DUP': 0, 'ALL': 0}
        }
        
        for key, count in raw_counts.items():
            # Parse the key to determine type and svtype
            if 'intersections/' in key or key.startswith('intersections'):
                category = 'intersections'
            elif 'unions/' in key or key.startswith('unions'):
                category = 'unions'
            else:
                continue  # Skip keys that don't match expected format
            
            # Determine svtype from the key
            if '.DEL.' in key or key.endswith('.DEL'):
                svtype = 'DEL'
            elif '.DUP.' in key or key.endswith('.DUP'):
                svtype = 'DUP'
            else:
                svtype = 'ALL'
            
            aggregated[category][svtype] += count
        
        post_results[input_set_name] = aggregated

    # Control sets - aggregate by svtype only
    for control_name in control_names:
        if control_name not in results or not results[control_name]:
            continue
        
        raw_counts = results[control_name]
        aggregated = {'DEL': 0, 'DUP': 0, 'ALL': 0}
        
        for key, count in raw_counts.items():
            # Determine svtype from the key
            if '.DEL' in key or key.endswith('.DEL'):
                aggregated['DEL'] += count
            elif '.DUP' in key or key.endswith('.DUP'):
                aggregated['DUP'] += count
        
        # Compute ALL as sum of DEL and DUP
        aggregated['ALL'] = aggregated['DEL'] + aggregated['DUP']
        post_results[control_name] = aggregated

    # Benchmark set - aggregate by svtype only
    if 'Benchmark' in results and results['Benchmark']:
        raw_counts = results['Benchmark']
        aggregated = {'DEL': 0, 'DUP': 0, 'ALL': 0}
        
        for key, count in raw_counts.items():
            # Determine svtype from the key (format: sample/sample.merged.DEL)
            if '.DEL' in key or key.endswith('.DEL'):
                aggregated['DEL'] += count
            elif '.DUP' in key or key.endswith('.DUP'):
                aggregated['DUP'] += count
        
        # Compute ALL as sum of DEL and DUP
        aggregated['ALL'] = aggregated['DEL'] + aggregated['DUP']
        post_results['Benchmark'] = aggregated

    return results, post_results


def get_samples_from_data(all_data: Dict[str, Dict[str, pd.DataFrame]], classification_key: str) -> set:
    """Extract sample names from a specific classification across all input sets."""

    all_samples = set()
    # Iterate only over input_sets (shared_FN is separate)
    for input_set_name, analysis_data in all_data['input_sets'].items():
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

def plot_excluded_regions_violin_plots(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create separate figures for excluded regions analysis.
    Each figure has a 2xN grid: Rows (DEL, DUP), Columns (one per input_set).
    Creates two figures: one for absolute change, one for percent change.
    
    Args:
        df: DataFrame with excluded regions data
        output_path: Path to save the figures (will append _abs_change.png and _pct_change.png)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get unique input sets and callers
    input_sets = sorted(df["input_set"].dropna().unique().tolist())
    callers = sorted(df["caller"].dropna().unique().tolist())
    
    num_input_sets = len(input_sets)
    num_callers = len(callers)
    
    # Color palette for callers
    colors = matplotlib.colormaps['Set3'](np.linspace(0, 1, num_callers))
    
    # Create figures for each metric
    for metric, metric_filename in [('abs_change', 'abs_change'), ('pct_change', 'pct_change')]:
        fig, axes = plt.subplots(2, num_input_sets, figsize=(6 * num_input_sets, 10))
        
        # Handle case with single input set
        if num_input_sets == 1:
            axes = axes.reshape(2, 1)
        
        for row_idx, svtype in enumerate(['DEL', 'DUP']):
            for col_idx, input_set in enumerate(input_sets):
                ax = axes[row_idx, col_idx]
                
                # Filter data for this svtype and input_set
                subset_df = df[(df['svtype'] == svtype) & (df['input_set'] == input_set)]
                
                if subset_df.empty:
                    ax.text(0.5, 0.5, f'No data', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(f'{svtype} - {input_set}')
                    continue
                
                # Plot violins for each caller
                for j, caller in enumerate(callers):
                    caller_data = subset_df[subset_df['caller'] == caller][metric].dropna()
                    data = caller_data.values
                    
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
                ax.set_title(f'{svtype} - {input_set}', fontsize=11, fontweight='bold')
                ax.set_xticks(range(1, num_callers + 1))
                ax.set_xticklabels(callers)
                ax.set_xlabel('Caller')
                
                # Set y-label
                if metric == 'abs_change':
                    ylabel_text = 'Absolute Change (# of calls removed)'
                else:
                    ylabel_text = 'Percent Change (%)'
                ax.set_ylabel(ylabel_text)
        
        # Set overall title
        metric_title = 'Absolute Change' if metric == 'abs_change' else 'Percent Change'
        fig.suptitle(f'Excluded Regions Filtering Impact - {metric_title}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save with metric-specific filename
        output_file = output_path.parent / f"{output_path.stem}_{metric_filename}.png"
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Saved {output_file.name}")

def plot_liftover_results(liftover_results_dict: Dict, output_dir: Path) -> None:
    """
    Create box plots for liftover success/failure rates.
    
    Args:
        liftover_results_dict: Dictionary with liftover results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    
    # Parse liftover results
    for input_set, input_data in liftover_results_dict.items():
        if 'samples' not in input_data or not isinstance(input_data['samples'], list):
            continue
        
        for sample_record in input_data['samples']:
            sample = sample_record.get('sample')
            svtype = sample_record.get('svtype')
            
            before = sample_record.get('record_count_before_liftover', 0)
            after = sample_record.get('record_count_after_liftover', 0)
            failed_liftover = sample_record.get('failed_liftover', 0)
            failed_size_change = sample_record.get('failed_size_change', 0)
            
            if before == 0:
                continue
            
            # Calculate percentages
            pct_succeeded = (after / before) * 100
            pct_failed_liftover = (failed_liftover / before) * 100
            pct_failed_size_change = (failed_size_change / before) * 100
            
            rows.append({
                'input_set': input_set,
                'sample': sample,
                'svtype': svtype,
                'pct_succeeded': pct_succeeded,
                'pct_failed_liftover': pct_failed_liftover,
                'pct_failed_size_change': pct_failed_size_change,
            })
    
    if not rows:
        print("No liftover data to plot")
        return
    
    df = pd.DataFrame(rows)
    print("\nLiftover Results Summary:")
    print(df.head())
    
    # Create side-by-side subplots for DEL and DUP
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    outcome_types = ['pct_succeeded', 'pct_failed_liftover', 'pct_failed_size_change']
    outcome_labels = ['Succeeded', 'Failed (Unmapped)', 'Failed (>10% Size Change)']
    colors = ['#2ecc71', '#e74c3c', '#f39c12']  # green, red, orange
    
    for svtype_idx, svtype in enumerate(['DEL', 'DUP']):
        ax = axes[svtype_idx]
        subset = df[df['svtype'] == svtype]
        
        if subset.empty:
            print(f"No liftover data for {svtype}")
            continue
        
        data_to_plot = []
        positions = []
        box_colors = []
        
        for i, (outcome_col, outcome_label) in enumerate(zip(outcome_types, outcome_labels)):
            data = subset[outcome_col].values
            data_to_plot.append(data)
            positions.append(i + 1)
            box_colors.append(colors[i])
        
        # Create box plots
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, 
                        patch_artist=True, showfliers=True, tick_labels=outcome_labels)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        # Overlay individual points with jitter
        for i, (data, pos) in enumerate(zip(data_to_plot, positions)):
            jitter = np.random.normal(loc=0, scale=0.08, size=len(data))
            ax.scatter(np.full(len(data), pos) + jitter, data, s=15, alpha=0.6, color='black', zorder=3)
        
        ax.set_title(f"{svtype} Records", fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(-5, 105)
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Liftover Results Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "liftover_results_boxplot.png", dpi=150)
    plt.close()
    print("Saved liftover_results_boxplot.png")

def analyze_logs(log_dir: Path, output_dir: Path, samples: Optional[List[str]] = None):

    # benchmark_processing_results.json
    # benchmark_merging_file = log_dir / "benchmark_processing_results.json"
    # benchmark_merging_df = pd.read_json(benchmark_merging_file).T
    # print("\nBenchmark Merging Results:")
    # print(benchmark_merging_df.head())

    # consensus_2of3_results.json
    # consensus_calls_file = log_dir / "consensus_2of3_results.json"
    # consensus_calls_dict = json.loads(consensus_calls_file.read_text())

    # changes_df = _compute_change_rows(consensus_calls_dict)
    
    # Filter by samples if provided
    # if samples is not None:
    #     sample_set = set(samples)
    #     original_count = len(changes_df)
    #     changes_df = changes_df[changes_df["sample"].isin(sample_set)].copy()
    #     filtered_count = len(changes_df)
    #     print(f"\nFiltered to {len(sample_set)} specified samples ({original_count - filtered_count} rows removed)")
    
    # print("\nConsensus Calls Change Summary:")
    # print(changes_df.head())

    # Group statistics by input_set and caller
    # grouped_means = (
    #     changes_df.groupby(["input_set", "caller"], dropna=True)[["abs_change", "pct_change"]]
    #     .mean()
    #     .reset_index()
    #     .sort_values(["input_set", "caller"])
    # )
    # print("\nMean Changes by Input Set and Caller:")
    # print(grouped_means)

    # figures_dir = output_dir / "figures"
    # figures_subdir = figures_dir / "excluded_regions_analysis"
    # figures_subdir.mkdir(parents=True, exist_ok=True)
    # plot_excluded_regions_violin_plots(
    #     changes_df,
    #     output_path=figures_subdir / "excluded_regions.png",
    # )

    # liftover_results.json
    # liftover_results_file = log_dir / "liftover_results.json"
    # if liftover_results_file.exists():
    #     liftover_results_dict = json.loads(liftover_results_file.read_text())
    #     plot_liftover_results(liftover_results_dict, figures_dir / "liftover_results")
    # else:
    #     print(f"Warning: Liftover results file not found: {liftover_results_file}")
    pass
