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

from load_analysis_data import build_analysis_data_structure, print_summary_statistics, filter_by_size


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


def plot_statistical_distribution(
    all_data: Dict[str, Dict[str, pd.DataFrame]], 
    metric_function: Callable[[int, int, int], float],
    start: float = 1e3,
    end: float = 1e6,
    n_points: int = 50,
    distribution_type: str = "histogram",
    svtype: Optional[str] = None,
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
    One line is plotted for each input set (e.g., "Low Coverage", "High Coverage").
    Applies Gaussian smoothing to reduce noise from sparse intervals.
    
    Args:
        all_data: Dictionary mapping input_set_name -> {classification -> dataframe}
        metric_function: Callable that takes (TP, FP, FN) counts and returns a metric value
        start: Starting size for intervals (default: 1kb)
        end: Ending size for intervals (default: 1Mb)
        n_points: Number of points to generate along the x-axis
        distribution_type: Type of size intervals ("histogram", "cumulative", "complementary_cumulative")
        svtype: Filter by specific svtype ('DEL', 'DUP') or None for all
        output_path: Path to save the figure (if None, displays instead)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as (width, height) tuple
        smoothing_sigma: Sigma parameter for Gaussian smoothing (0 = no smoothing)
        show_raw_points: Whether to show raw data points faintly beneath smoothed line
        min_samples: Minimum number of total samples (TP+FP+FN) required to include a point
    
    Example:
        >>> def precision(tp, fp, fn):
        ...     return tp / (tp + fp) if (tp + fp) > 0 else 0
        >>> plot_statistical_distribution(all_data, precision, ylabel="Precision", smoothing_sigma=5)
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Generate size intervals
    intervals = generate_size_intervals(start, end, n_points, distribution_type)
    
    # Collect data by input set
    data_by_input = {}
    
    for input_set_name, analysis_data in all_data.items():
        x_values = []
        metric_values = []
        total_samples = []
        
        for lower, upper in intervals:
            # Filter data by size
            filtered = filter_by_size(analysis_data, lower_bound=int(lower), upper_bound=int(upper))
            
            # Apply svtype filter if specified
            if svtype is not None:
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
                # Count all records
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
        
        # Store data for this input set
        data_by_input[input_set_name] = {
            'x': np.array(x_values),
            'y': np.array(metric_values),
            'samples': np.array(total_samples)
        }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each input set
    cmap = colormaps["tab10"]
    colors = [cmap(i) for i in range(len(data_by_input))]
    
    for idx, (input_set_name, data) in enumerate(data_by_input.items()):
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
        
        color = colors[idx]
        
        # Plot smoothed line
        ax.plot(
            x_sorted,
            y_smoothed,
            label=input_set_name,
            color=color,
            linewidth=2.5,
            alpha=0.9
        )
        
        # Optionally show raw data points
        if show_raw_points:
            ax.scatter(
                x_sorted,
                y_sorted,
                color=color,
                alpha=0.2,
                s=20,
                zorder=2
            )
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Input Set', fontsize=10, title_fontsize=11)
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
    
    # Example: Plot precision distribution
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    
    output_dir = Path(config['output_dir']) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_statistical_distribution(
        all_data,
        metric_function=precision,
        start=1e2,
        end=1e6,
        n_points=100,  # High resolution
        distribution_type="histogram",
        svtype=None,  # All types
        output_path=str(output_dir / "precision_by_size.png"),
        title="Precision by CNV Size (All Types)",
        ylabel="Precision",
        smoothing_sigma=5.0,  # Apply smoothing
        show_raw_points=True,  # Show underlying data
        min_samples=5  # Filter sparse intervals
    )
    
    # Plot recall for DEL only
    plot_statistical_distribution(
        all_data,
        metric_function=recall,
        start=1e2,
        end=1e6,
        n_points=100,
        distribution_type="histogram",
        svtype="DEL",
        output_path=str(output_dir / "recall_by_size_DEL.png"),
        title="Recall by CNV Size (Deletions)",
        ylabel="Recall/Sensitivity",
        smoothing_sigma=5.0,
        show_raw_points=True,
        min_samples=5
    )
    
    # Plot F1 score for DUP only
    plot_statistical_distribution(
        all_data,
        metric_function=f1_score,
        start=1e2,
        end=1e6,
        n_points=100,
        distribution_type="histogram",
        svtype="DUP",
        output_path=str(output_dir / "f1_score_by_size_DUP.png"),
        title="F1 Score by CNV Size (Duplications)",
        ylabel="F1 Score",
        smoothing_sigma=5.0,
        show_raw_points=True,
        min_samples=5
    )
    
    print(f"\nPlots saved to: {output_dir}")
    print("="*80)

    # Get precision of high coverage input set for CNVs between 1kb and 2kb
    high_cov_data = all_data.get("High Coverage", {})
    if high_cov_data:
        filtered = filter_by_size(high_cov_data, lower_bound=1000, upper_bound=1500)
        tp_count = len(filtered.get('TP', pd.DataFrame()))
        fp_count = len(filtered.get('FP', pd.DataFrame()))
        fn_count = len(filtered.get('FN', pd.DataFrame()))
        precision_value = precision(tp_count, fp_count, fn_count)
        print(f"\nPrecision for High Coverage (1kb-2kb): {precision_value:.4f}")
    

if __name__ == "__main__":
    main()
