from utils import parse_args
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Callable, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from load_analysis_data import build_analysis_data_structure, print_summary_statistics, filter_by_size

# Test Functions:
def _test1(all_data):
    # Get precision of high coverage input set for CNVs between 1kb and 2kb
    high_cov_data = all_data.get("High Coverage", {})
    if high_cov_data:
        filtered = filter_by_size(high_cov_data, lower_bound=1000, upper_bound=1500)
        tp_count = len(filtered.get('TP', pd.DataFrame()))
        fp_count = len(filtered.get('FP', pd.DataFrame()))
        fn_count = len(filtered.get('FN', pd.DataFrame()))
        precision_value = precision(tp_count, fp_count, fn_count)
        print(f"\nPrecision for High Coverage (1kb-2kb): {precision_value:.4f}")

# Define metric functions
def precision(tp, fp, fn):
    """Calculate precision: TP / (TP + FP)"""
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp, fp, fn):
    """Calculate recall/sensitivity: TP / (TP + FN)"""
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(tp, fp, fn):
    """Calculate F1 score: 2 * (precision * recall) / (precision + recall)"""
    return (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0


def generate_size_intervals(
    start: float, 
    end: float, 
    n_points: int, 
    distribution_type: str = "histogram"
) -> List[Tuple[float, float]]:
    """
    Generate size intervals for different distribution analyses.
    
    Creates logarithmically-spaced points and generates intervals based on the
    distribution type:
    - histogram: Adjacent pairs (bin intervals)
    - cumulative: Start value to each point (cumulative from beginning)
    - complementary_cumulative: Each point to end value (cumulative from end)
    
    Args:
        start: Starting value (lower bound)
        end: Ending value (upper bound)
        n_points: Number of points in logspace
        distribution_type: Type of intervals to generate
            - "histogram": [(p0, p1), (p1, p2), ..., (pn-1, pn)]
            - "cumulative": [(start, p0), (start, p1), ..., (start, pn)]
            - "complementary_cumulative": [(p0, end), (p1, end), ..., (pn, end)]
    
    Returns:
        List of (lower, upper) tuples representing size intervals
    
    Examples:
        >>> generate_size_intervals(1e3, 1e6, 10, "histogram")
        [(1000, 2154), (2154, 4642), ..., (464159, 1000000)]
        
        >>> generate_size_intervals(1e3, 1e6, 10, "cumulative")
        [(1000, 1000), (1000, 2154), ..., (1000, 1000000)]
        
        >>> generate_size_intervals(1e3, 1e6, 10, "complementary_cumulative")
        [(1000, 1000000), (2154, 1000000), ..., (1000000, 1000000)]
    """
    # Generate logarithmically-spaced points
    points = np.logspace(np.log10(start), np.log10(end), n_points)
    
    intervals = []
    
    if distribution_type == "histogram":
        # Adjacent pairs: bin intervals for histogram
        for i in range(len(points) - 1):
            intervals.append((points[i], points[i + 1]))
    
    elif distribution_type == "cumulative":
        # Start to each point: cumulative distribution
        for point in points:
            intervals.append((start, point))
        # Remove the first interval if it is (start, start) to avoid zero-length interval
        if intervals and intervals[0][0] >= intervals[0][1]:
            intervals.pop(0)
    
    elif distribution_type == "complementary_cumulative":
        # Each point to end: complementary cumulative distribution
        for point in points:
            intervals.append((point, end))
        # Remove the last interval if it is (end, end) to avoid zero-length interval
        if intervals and intervals[-1][0] >= intervals[-1][1]:
            intervals.pop()
    
    else:
        raise ValueError(
            f"Unknown distribution_type: '{distribution_type}'. "
            f"Must be 'histogram', 'cumulative', or 'complementary_cumulative'"
        )
    
    return intervals

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
        filtered_data = filter_by_size(analysis_data, lower_bound=100, upper_bound=1_000_000)
        all_data[input_set_key] = filtered_data

    print("\nSummary of loaded data:")

    for input_set_key, analysis_data in all_data.items():
        print("\ninput_set_key:", input_set_key)
        print("Keys in analysis_data:", analysis_data.keys())
        for classification_key, df in analysis_data.items():
            print(f"  Classification: {classification_key}, Number of records: {len(df)}")
    
    # Example: Plot precision distribution
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

    # _test1(all_data)

    

if __name__ == "__main__":
    main()

