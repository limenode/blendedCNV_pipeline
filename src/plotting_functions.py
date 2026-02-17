from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from typing import List, Tuple, Callable, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from load_analysis_data import build_analysis_data_structure, print_summary_statistics, filter_by_size
from utils import generate_size_intervals, precision, recall, f1_score

# TODO: Filter by binned intervals and merge intervals for cumulative distributions for better performance.
def plot_statistical_distribution(
    all_data: Dict[str, Dict[str, pd.DataFrame]], 
    metric_function: Callable[[int, int, int], float],
    start: float = 1e3,
    end: float = 1e6,
    n_points: int = 50,
    distribution_type: str = "histogram",
    svtypes: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: str = "CNV Performance by Size",
    xlabel: str = "CNV Size (bp)",
    ylabel: str = "Metric Value",
    figsize: Tuple[int, int] = (12, 6),
    smoothing_sigma: float = 5.0,
    show_raw_points: bool = True,
    min_samples: int = 0
):
    """
    Plot a statistical distribution of CNV metrics across size ranges with smoothing.
    
    Creates a plot with CNV size on the x-axis and a computed metric on the y-axis.
    Multiple lines are plotted for each combination of input set and svtype.
    Applies Gaussian smoothing to reduce noise from sparse intervals.
    
    Args:
        all_data: Dictionary mapping input_set_name -> {classification -> dataframe}
        metric_function: Callable that takes (TP, FP, FN) counts and returns a metric value
        start: Starting size for intervals (default: 1kb)
        end: Ending size for intervals (default: 1Mb)
        n_points: Number of points to generate along the x-axis
        distribution_type: Type of size intervals ("histogram", "cumulative", "complementary_cumulative")
        svtypes: List of svtypes to plot (['ALL', 'DEL', 'DUP']) or None for ['ALL'] only
        output_path: Path to save the figure (if None, displays instead)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as (width, height) tuple
        smoothing_sigma: Sigma parameter for Gaussian smoothing (0 = no smoothing)
        show_raw_points: Whether to show raw data points faintly beneath smoothed line. Only shown if svtypes is ['ALL'] or if there is only one svtype to avoid clutter.
        min_samples: Minimum number of total samples (TP+FP+FN) required to include a point
    
    Example:
        >>> def precision(tp, fp, fn):
        ...     return tp / (tp + fp) if (tp + fp) > 0 else 0
        >>> plot_statistical_distribution(all_data, precision, svtypes=['ALL', 'DEL', 'DUP'], ylabel="Precision")
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Default to ['ALL'] if svtypes not specified
    if svtypes is None:
        svtypes = ['ALL']
    
    # Generate size intervals
    intervals = generate_size_intervals(start, end, n_points, distribution_type)
    
    # Collect data by input set and svtype combination
    data_by_combination = {}
    
    for input_set_name, analysis_data in all_data.items():
        for svtype in svtypes:
            x_values = []
            metric_values = []
            total_samples = []
            
            for lower, upper in intervals:
                # Filter data by size
                filtered = filter_by_size(analysis_data, lower_bound=int(lower), upper_bound=int(upper))
                
                # Apply svtype filter if not 'ALL'
                if svtype != 'ALL':
                    tp_df = filtered.get('TP', pd.DataFrame())
                    fp_df = filtered.get('FP', pd.DataFrame())
                    fn_df = filtered.get('FN', pd.DataFrame())
                    
                    if not tp_df.empty:
                        tp_df = tp_df[tp_df['svtype'] == svtype]
                    if not fp_df.empty:
                        fp_df = fp_df[fp_df['svtype'] == svtype]
                    if not fn_df.empty:
                        fn_df = fn_df[fn_df['svtype'] == svtype]
                    
                    tp_count = len(tp_df)
                    fp_count = len(fp_df)
                    fn_count = len(fn_df)
                else:
                    # Count all records for 'ALL'
                    tp_count = len(filtered.get('TP', pd.DataFrame()))
                    fp_count = len(filtered.get('FP', pd.DataFrame()))
                    fn_count = len(filtered.get('FN', pd.DataFrame()))
                
                total = tp_count + fp_count + fn_count
                
                # Skip if below minimum sample threshold
                if total < min_samples:
                    continue
                
                # Calculate metric using the provided function
                metric_value = metric_function(tp_count, fp_count, fn_count)
                
                # Use geometric mean for x-axis on log scale (histogram mode)
                if distribution_type == "histogram":
                    x_value = np.sqrt(lower * upper)
                elif distribution_type == "cumulative":
                    x_value = upper
                elif distribution_type == "complementary_cumulative":
                    x_value = lower
                else:
                    x_value = (lower + upper) / 2
                
                x_values.append(x_value)
                metric_values.append(metric_value)
                total_samples.append(total)
            
            # Store data for this combination
            data_by_combination[(input_set_name, svtype)] = {
                'x': np.array(x_values),
                'y': np.array(metric_values),
                'samples': np.array(total_samples)
            }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for input sets and line styles for svtypes
    input_set_names = list(all_data.keys())
    cmap = colormaps["tab10"]
    color_map = {name: cmap(i) for i, name in enumerate(input_set_names)}
    
    # Line styles for different svtypes
    linestyle_map = {
        'ALL': '-',      # solid
        'DEL': '--',     # dashed
        'DUP': ':',      # dotted
    }
    
    # Plot each combination
    for (input_set_name, svtype), data in data_by_combination.items():
        if len(data['x']) == 0:
            continue
        
        # Sort by x-axis values
        sort_idx = np.argsort(data['x'])
        x_sorted = data['x'][sort_idx]
        y_sorted = data['y'][sort_idx]
        
        # Apply Gaussian smoothing if sigma > 0
        if smoothing_sigma > 0 and len(y_sorted) > 1:
            y_smoothed = gaussian_filter1d(y_sorted, sigma=smoothing_sigma)
        else:
            y_smoothed = y_sorted
        
        color = color_map[input_set_name]
        linestyle = linestyle_map.get(svtype, '-')
        
        # Create label
        label = f"{input_set_name} - {svtype}"
        
        # If "ALL" is in svtypes and len(svtypes) > 1, reduce alpha of others to highlight "ALL"
        if 'ALL' in svtypes and len(svtypes) > 1:
            if svtype == 'ALL':
                alpha = 0.9
                linewidth = 3.0
            else:
                alpha = 0.45
                linewidth = 2.0
        else:
            alpha = 0.9
            linewidth = 2.5

        # Plot smoothed line
        ax.plot(
            x_sorted,
            y_smoothed,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha
        )
        
        # Optionally show raw data points
        if show_raw_points and (len(svtypes) == 1 or svtype == 'ALL'):
            ax.scatter(
                x_sorted,
                y_sorted,
                color=color,
                alpha=0.15,
                s=15,
                zorder=2
            )
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, title_fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_venn_diagram(
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    input_set_keys: List[str],
    svtype: Optional[str] = None,
    input_set_name_map: Optional[Dict[str, str]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a Venn diagram showing overlap of benchmark records recalled by different methods.
    
    Reads true positives from each input set and creates a Venn diagram showing which
    benchmark CNVs are detected by each method, including overlaps. Benchmark records are
    identified by their (truth_chrom, truth_start, truth_end, svtype) tuple.
    
    Args:
        all_data: Dictionary mapping input_set_name -> {classification -> dataframe}
        input_set_keys: List of 2-3 input set names to compare (must exist in all_data)
        svtype: Optional filter for specific svtype ('DEL', 'DUP', or None for all)
        input_set_name_map: Optional mapping for display names
        title: Plot title (auto-generated if empty)
        figsize: Figure size as (width, height) tuple
        output_path: Path to save the figure (if None, displays instead)
    
    Returns:
        DataFrame with detection information for each benchmark record
    
    Example:
        >>> plot_venn_diagram(
        ...     all_data,
        ...     input_set_keys=['High_Coverage_intersections', 'Low_Coverage_intersections'],
        ...     svtype='DEL',
        ...     title="Deletion Detection Overlap"
        ... )
    """
    if len(input_set_keys) < 2 or len(input_set_keys) > 3:
        raise ValueError("Venn diagram requires 2 or 3 input sets")
    
    print(len(input_set_keys), "input sets provided for Venn diagram:", input_set_keys)

    # Verify all keys exist in all_data
    for key in input_set_keys:
        if key not in all_data:
            raise ValueError(f"Input set key '{key}' not found in all_data")
    
    # Required columns for building benchmark IDs
    required_cols = ['truth_chrom', 'truth_start', 'truth_end', 'svtype']
    
    # Extract TP DataFrames and create sets of benchmark IDs
    tp_sets = {}
    
    for input_set_key in input_set_keys:
        tp_df = all_data[input_set_key].get('TP', pd.DataFrame())
        
        if tp_df.empty:
            print(f"Warning: No true positives found for '{input_set_key}'")
            tp_sets[input_set_key] = set()
            continue
        
        # Verify required columns exist
        missing_cols = [col for col in required_cols if col not in tp_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns {missing_cols} in TP DataFrame for '{input_set_key}'. "
                f"Available columns: {list(tp_df.columns)}"
            )
        
        # Apply svtype filter if specified
        if svtype and 'svtype' in tp_df.columns:
            tp_df = tp_df[tp_df['svtype'] == svtype].copy()
        
        # Build benchmark IDs as tuples (chrom, start, end, svtype)
        # Using tuples for fast comparison - start position is first for quick filtering
        benchmark_ids = set(
            tp_df[required_cols].itertuples(index=False, name=None)
        )
        tp_sets[input_set_key] = benchmark_ids
    
    # Calculate universal set (all benchmark IDs across all input sets)
    all_benchmark_ids = set().union(*tp_sets.values())
    total_unique_cnvs = len(all_benchmark_ids)
    
    # Create detection summary DataFrame
    detection_records = []
    for bench_id_tuple in all_benchmark_ids:
        detected_by = [set_key for set_key, tp_set in tp_sets.items() if bench_id_tuple in tp_set]
        detection_records.append({
            'truth_chrom': bench_id_tuple[0],
            'truth_start': bench_id_tuple[1],
            'truth_end': bench_id_tuple[2],
            'svtype': bench_id_tuple[3],
            'detected_by': detected_by,
            'detection_count': len(detected_by)
        })
    
    detection_df = pd.DataFrame(detection_records)
    
    # Calculate statistics
    total_detected_by_at_least_one = len(detection_df[detection_df['detection_count'] > 0])
    
    # Apply name mapping and add counts to labels
    display_names_with_counts = []
    for input_set_key in input_set_keys:
        count = len(tp_sets[input_set_key])
        pct_total = (count / total_unique_cnvs) * 100 if total_unique_cnvs > 0 else 0
        
        if input_set_name_map:
            display_name = input_set_name_map.get(input_set_key, input_set_key)
        else:
            display_name = input_set_key
        
        label = f"{display_name}\n(n={count}, {pct_total:.1f}%)"
        display_names_with_counts.append(label)
    
    # Create Venn diagram
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(input_set_keys) == 2:
        display_names_2: Tuple[str, str] = (display_names_with_counts[0], display_names_with_counts[1])
        venn_obj = venn2(
            subsets=tuple(tp_sets.values()),
            set_labels=display_names_2,
            ax=ax
        )
        venn_circles_obj = venn2_circles(subsets=tuple(tp_sets.values()), ax=ax)
    else:  # 3 input sets
        display_names_3: Tuple[str, str, str] = (
            display_names_with_counts[0], 
            display_names_with_counts[1], 
            display_names_with_counts[2]
        )
        venn_obj = venn3(
            subsets=tuple(tp_sets.values()),
            set_labels=display_names_3,
            ax=ax
        )
        venn_circles_obj = venn3_circles(subsets=tuple(tp_sets.values()), ax=ax)
    
    # Customize appearance
    for circle in venn_circles_obj:
        circle.set_linewidth(2)
        circle.set_linestyle('--')
    
    # Generate title
    if not title:
        svtype_str = f" ({svtype})" if svtype else ""
        title = f"Benchmark CNV Recall Overlap{svtype_str}\n({total_unique_cnvs} total unique benchmark records)"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add summary statistics text box
    stats_text = (
        f"Total unique CNVs: {total_unique_cnvs} | "
        f"Detected by ≥1: {total_detected_by_at_least_one} "
        f"({total_detected_by_at_least_one/total_unique_cnvs*100:.1f}%)"
    )
    ax.text(
        0.5, -0.15, stats_text, 
        ha='center', 
        transform=ax.transAxes,
        fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Venn diagram saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print detailed statistics
    print(f"\n{'='*60}")
    print(f"Venn Diagram Detection Statistics")
    print(f"{'='*60}")
    print(f"Total unique benchmark CNVs: {total_unique_cnvs}")
    print(f"Detected by at least one method: {total_detected_by_at_least_one} "
          f"({total_detected_by_at_least_one/total_unique_cnvs*100:.1f}%)")
    
    print(f"\nDetection by individual methods:")
    for input_set_key in input_set_keys:
        count = len(tp_sets[input_set_key])
        display_name = input_set_name_map.get(input_set_key, input_set_key) if input_set_name_map else input_set_key
        pct_total = (count / total_unique_cnvs) * 100 if total_unique_cnvs > 0 else 0
        print(f"  {display_name}: {count} ({pct_total:.1f}% of total)")
    
    # Print detailed overlap counts
    print(f"\nDetailed Overlap Counts:")
    subsets = venn_obj.get_label_by_id
    if len(input_set_keys) == 2:
        combinations = ['10', '01', '11']
    else:
        combinations = ['100', '010', '001', '110', '101', '011', '111']
    
    for comb in combinations:
        label = subsets(comb)
        if label is None:
            count = 0
        else:
            count = int(label.get_text())
        
        pct_total = (count / total_unique_cnvs) * 100 if total_unique_cnvs > 0 else 0
        
        # Calculate % of each method involved in this combination
        involved_indices = [i for i, bit in enumerate(comb) if bit == '1']
        pct_each_method = []
        for i in involved_indices:
            method_count = len(tp_sets[input_set_keys[i]])
            pct_method = (count / method_count) * 100 if method_count > 0 else 0
            method_name = input_set_name_map.get(input_set_keys[i], input_set_keys[i]) if input_set_name_map else input_set_keys[i]
            pct_each_method.append(f"{method_name}: {pct_method:.1f}%")
        pct_each_method_str = "; ".join(pct_each_method)
        
        print(f"  Combination {comb}: {count} ({pct_total:.1f}% of total) | {pct_each_method_str}")
    
    return detection_df

def _generate_single_plot(args_tuple):
    """
    Worker function for multiprocessing plot generation.
    
    Args:
        args_tuple: Tuple of (all_data, dist_type, svtypes, metric_func, ylabel, 
                              start, end, npoints, output_dir, smoothing_sigma, 
                              show_raw_points, min_samples)
    
    Returns:
        String indicating success or failure
    """
    (all_data, dist_type, svtypes, metric_func, ylabel, 
     start, end, npoints, output_dir, smoothing_sigma, 
     show_raw_points, min_samples) = args_tuple
    
    # Get metric name from function
    metric_name = metric_func.__name__
    
    # Create filename based on svtypes list
    svtypes_str = "_".join(svtypes) if svtypes else "all"
    filename = f"{metric_name}_{dist_type}_{svtypes_str}.png"
    
    # Create title
    if len(svtypes) == 1:
        svtype_title = f"({svtypes[0]})"
    else:
        svtype_title = f"({', '.join(svtypes)})"
    dist_type_name = dist_type.replace('_', ' ').title()
    title = f"{ylabel} by CNV Size - {dist_type_name} {svtype_title}"
    
    output_path = str(Path(output_dir) / filename)
    
    try:
        plot_statistical_distribution(
            all_data,
            metric_function=metric_func,
            start=start,
            end=end,
            n_points=npoints,
            distribution_type=dist_type,
            svtypes=svtypes,
            output_path=output_path,
            title=title,
            ylabel=ylabel,
            smoothing_sigma=smoothing_sigma,
            show_raw_points=show_raw_points,
            min_samples=min_samples
        )
        return f"✓ {filename}"
    except Exception as e:
        return f"✗ {filename}: {str(e)}"
    
def plot_performance_distributions(config: dict, all_data: Dict[str, Dict[str, pd.DataFrame]]):
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    
    output_dir = Path(config['output_dir']) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = 100
    end = 1_000_000
    npoints = 100

    distribution_types = ["histogram", "cumulative", "complementary_cumulative"]
    # Define svtype combinations to plot
    svtype_combinations = [
        # ['ALL'],              # Just aggregate
        # ['DEL'],              # Just deletions  
        # ['DUP'],              # Just duplications
        ['ALL', 'DEL', 'DUP'] # All three on one plot for comparison
    ]
    metrics = [(precision, 'Precision'), (recall, 'Recall/Sensitivity'), (f1_score, 'F1 Score')]

    smoothing_sigma = 5.0
    min_samples = 5
    show_raw_points = True

    # Create all combinations of (distribution_type, svtypes, metric_func, ylabel)
    plot_configs = []
    for dist_type in distribution_types:
        for svtypes in svtype_combinations:
            for (metric_func, ylabel) in metrics:
                plot_configs.append((dist_type, svtypes, metric_func, ylabel))
    
    # Prepare arguments for multiprocessing
    # Each tuple contains all the data needed for one plot
    plot_args = []
    for dist_type, svtypes, metric_func, ylabel in plot_configs:
        args_tuple = (
            all_data, dist_type, svtypes, metric_func, ylabel,
            start, end, npoints, output_dir, smoothing_sigma,
            show_raw_points, min_samples
        )
        plot_args.append(args_tuple)
    
    # Generate all plots in parallel using multiprocessing
    print(f"\nGenerating {len(plot_configs)} plots using multiprocessing ({len(plot_configs)} workers)...")
    
    with ProcessPoolExecutor() as executor:
        # Submit all plot generation tasks
        futures = [executor.submit(_generate_single_plot, args) for args in plot_args]
        
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            print(result)
    
    print(f"\n{'='*80}")
    print(f"All plots completed! Saved to: {output_dir}")
    print("="*80)

def plot_venn_diagram_wrapper(config: dict, all_data: Dict[str, Dict[str, pd.DataFrame]]):
    print("\n" + "="*80)
    print("Generating Venn diagrams...")
    print("="*80)
    
    venn_output_dir = Path(config['output_dir']) / "venn_diagrams"
    venn_output_dir.mkdir(parents=True, exist_ok=True)
    
    keys_of_interest = ["Low_Coverage_intersections", "High_Coverage_intersections", "SNP_Array"]
    if len(keys_of_interest) >= 2:
        input_set_name_map = {key: key.replace('_', ' ').title() for key in keys_of_interest}
        venn_output_path = venn_output_dir / "venn_diagram.png"
        
        detection_df = plot_venn_diagram(
            all_data=all_data,
            input_set_keys=keys_of_interest,
            svtype=None,  # All svtypes
            input_set_name_map=input_set_name_map,
            title="CNV Detection Overlap Across Methods",
            output_path=str(venn_output_path)
        )
        print(f"\n✓ Venn diagram saved. Detection DataFrame shape: {detection_df.shape}")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print("="*80)

def plot_size_distribution(
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    input_set_keys: List[str],
    svtype: Optional[str] = None,
    input_set_name_map: Optional[Dict[str, str]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[str | Path] = None
):
    """
    Plot the size distribution of detected CNVs for different input sets and svtypes.
    
    Creates two separate plots:
    1. Histogram with square bins showing count distributions
    2. KDE (Kernel Density Estimate) plot showing smooth density curves
    
    Compares predicted CNVs (TP + FP) from each input set against the benchmark truth set (TP + FN).
    
    Args:
        all_data: Dictionary mapping input_set_name -> {classification -> dataframe}
        input_set_keys: List of input set names to include in the plot
        svtype: Optional filter for specific svtype ('DEL', 'DUP', or None for all)
        input_set_name_map: Optional mapping for display names
        title: Base title for plots (will be suffixed with plot type)
        figsize: Figure size as (width, height) tuple
        output_path: Base path to save figures. Will append '_histogram.png' and '_kde.png'
                    If None, displays plots instead
    
    Example:
        >>> plot_size_distribution(
        ...     all_data,
        ...     input_set_keys=['High_Coverage_intersections', 'Low_Coverage_intersections'],
        ...     svtype='DEL',
        ...     title="Deletion Size Distribution"
        ... )
    """
    # Validate input
    if not input_set_keys:
        raise ValueError("At least one input_set_key must be provided")
    
    for key in input_set_keys:
        if key not in all_data:
            raise ValueError(f"Input set key '{key}' not found in all_data")
    
    # Collect all predicted records from TP + FP for the specified input sets and svtype
    datasets = defaultdict(dict)
    for input_set_key in input_set_keys:
        predicted_records = []
        for classification in ['TP', 'FP']:
            df = all_data[input_set_key].get(classification, pd.DataFrame())
            if df.empty:
                continue
            
            if svtype and 'svtype' in df.columns:
                df = df[df['svtype'] == svtype].copy()
            
            predicted_records.append(df)
        
        if predicted_records:
            combined_df = pd.concat(predicted_records, ignore_index=True)
            datasets['pred'][input_set_key] = combined_df
    
    # Collect all benchmark records from the TP + FN of the input sets (e.g., the first one) for the specified svtype
    benchmark_records = []
    for classification in ['TP', 'FN']:
        df = all_data[input_set_keys[0]].get(classification, pd.DataFrame())
        if df.empty:
            continue
        
        if svtype and 'svtype' in df.columns:
            df = df[df['svtype'] == svtype].copy()
        
        benchmark_records.append(df)
    
    if benchmark_records:
        benchmark_df = pd.concat(benchmark_records, ignore_index=True)
        datasets['truth']['Benchmark'] = benchmark_df
    
    # Set up color palette
    cmap = colormaps["tab10"]
    color_idx = 0
    
    # Prepare output paths
    if output_path:
        output_base = Path(output_path)
        output_dir = output_base.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_base.stem
        suffix = output_base.suffix
        histogram_path = output_dir / f"{stem}_histogram{suffix}"
        kde_path = output_dir / f"{stem}_kde{suffix}"
    else:
        histogram_path = None
        kde_path = None
    
    # ==================== PLOT 1: HISTOGRAM WITH BINS ====================
    fig, ax = plt.subplots(figsize=figsize)
    
    # First pass: collect all size data to determine common bin range
    all_size_data = []
    valid_datasets = []
    
    for size_type, included_sets_dict in datasets.items():
        for included_set_name, df in included_sets_dict.items():
            if f'{size_type}_size' not in df.columns:
                print(f"Warning: '{size_type}_size' column not found in data for '{included_set_name}'. Skipping.")
                continue
            
            # Get size data and filter out invalid values
            size_data = df[f'{size_type}_size'].dropna()
            size_data = size_data[size_data > 0]  # Remove non-positive values for log scale
            
            if len(size_data) == 0:
                print(f"Warning: No valid size data for '{included_set_name}'. Skipping.")
                continue
            
            all_size_data.extend(size_data.values)
            valid_datasets.append((size_type, included_set_name, size_data))
    
    # Create common logarithmically spaced bins across all datasets
    if len(all_size_data) > 0:
        log_min = np.log10(min(all_size_data))
        log_max = np.log10(max(all_size_data))
        common_bins = np.logspace(log_min, log_max, 51)  # 50 bins
        
        # Second pass: plot all histograms with common bins
        for size_type, included_set_name, size_data in valid_datasets:
            label = input_set_name_map.get(included_set_name, included_set_name) if input_set_name_map else included_set_name
            color = cmap(color_idx % 10)
            
            # Plot histogram with step style using common bins
            ax.hist(
                size_data,
                bins=common_bins,
                density=True,
                histtype='step',
                label=label,
                color=color,
                linewidth=2,
                alpha=0.8
            )
            color_idx += 1
    
    ax.set_xlabel("Size (bp)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xscale('log')
    histogram_title = f"{title} - Histogram" if title else "CNV Size Distribution - Histogram"
    ax.set_title(histogram_title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if histogram_path:
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        print(f"✓ Histogram plot saved to: {histogram_path}")
    else:
        plt.show()
    
    plt.close()
    
    # ==================== PLOT 2: KDE CURVES ====================
    fig, ax = plt.subplots(figsize=figsize)
    color_idx = 0
    
    for size_type, included_sets_dict in datasets.items():
        for included_set_name, df in included_sets_dict.items():
            if f'{size_type}_size' not in df.columns:
                continue
            
            label = input_set_name_map.get(included_set_name, included_set_name) if input_set_name_map else included_set_name
            color = cmap(color_idx % 10)
            
            # Get size data and filter out invalid values
            size_data = df[f'{size_type}_size'].dropna()
            size_data = size_data[size_data > 0]  # Remove non-positive values for log scale
            
            if len(size_data) < 2:
                print(f"Warning: Insufficient data points for KDE for '{included_set_name}'. Skipping.")
                color_idx += 1
                continue
            
            # Compute KDE on log-transformed data for better visualization
            log_data = np.log10(size_data)
            
            try:
                kde = gaussian_kde(log_data)
                
                # Create evaluation points in log space
                log_min = log_data.min()
                log_max = log_data.max()
                log_range = log_max - log_min
                log_eval = np.linspace(log_min - 0.1 * log_range, log_max + 0.1 * log_range, 500)
                
                # Evaluate KDE
                density = kde(log_eval)
                
                # Transform back to linear space for plotting
                eval_points = 10 ** log_eval
                
                # Plot KDE curve
                ax.plot(
                    eval_points,
                    density,
                    label=label,
                    color=color,
                    linewidth=2.5,
                    alpha=0.8
                )
            except Exception as e:
                print(f"Warning: Could not compute KDE for '{included_set_name}': {e}")
            
            color_idx += 1
    
    ax.set_xlabel("Size (bp)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xscale('log')
    kde_title = f"{title} - KDE" if title else "CNV Size Distribution - KDE"
    ax.set_title(kde_title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if kde_path:
        plt.savefig(kde_path, dpi=300, bbox_inches='tight')
        print(f"✓ KDE plot saved to: {kde_path}")
    else:
        plt.show()
    
    plt.close()