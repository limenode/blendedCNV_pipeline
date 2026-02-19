from collections import defaultdict
from typing import Callable, Tuple, Optional, List, Dict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib_venn import venn3, venn3_circles
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns

from load_analysis_data import filter_by_size
from utils import generate_size_intervals, DistributionType, SVType


import time

# Module-level helper function for multiprocessing
def _create_record_ids(df, classification):
    """
    Create unique identifiers for records based on classification type.
    
    Args:
        df: DataFrame with records
        classification: One of 'TP', 'FP', or 'FN'
    
    Returns:
        Set of tuples uniquely identifying each record
    """
    if df.empty:
        return set()
    
    if classification in ['FP']:
        # Use predicted coordinates for FP
        required_cols = ['pred_chrom', 'pred_start', 'pred_end', 'svtype']
        if all(col in df.columns for col in required_cols):
            return set(df[required_cols].itertuples(index=False, name=None))
    elif classification in ['TP', 'FN']:
        # Use truth coordinates for TP and FN
        required_cols = ['truth_chrom', 'truth_start', 'truth_end', 'svtype']
        if all(col in df.columns for col in required_cols):
            return set(df[required_cols].itertuples(index=False, name=None))
    
    # Fallback: use row hashes if required columns not available
    return set(df.apply(lambda row: hash(tuple(row)), axis=1))


def _process_input_sv_combination_worker(args):
    """
    Worker function to process a single (input_set_name, svtype) combination.
    Designed to be called from multiprocessing pool.
    
    Args:
        args: Tuple of (input_set_name, svtype, analysis_data, intervals)
    
    Returns:
        Tuple of (key, result_dict) where key is (input_set_name, svtype)
        and result_dict contains data for all three distribution types
    """
    input_set_name, svtype, analysis_data, intervals = args
    
    # Store record IDs for each interval and classification
    interval_data = []
    
    # Pass 1: Filter by size intervals and create unique record IDs
    for lower, upper in intervals:
        # Filter data by size once for this interval
        filtered = filter_by_size(analysis_data, lower_bound=int(lower), upper_bound=int(upper))
        
        # Get classification DataFrames
        tp_df = filtered.get('TP', pd.DataFrame())
        fp_df = filtered.get('FP', pd.DataFrame())
        fn_df = filtered.get('FN', pd.DataFrame())
        
        # Apply svtype filter if not 'ALL'
        if svtype != SVType.ALL:
            if not tp_df.empty and 'svtype' in tp_df.columns:
                tp_df = tp_df[tp_df['svtype'] == svtype.value].copy()
            if not fp_df.empty and 'svtype' in fp_df.columns:
                fp_df = fp_df[fp_df['svtype'] == svtype.value].copy()
            if not fn_df.empty and 'svtype' in fn_df.columns:
                fn_df = fn_df[fn_df['svtype'] == svtype.value].copy()
        
        # Create unique IDs for each classification
        tp_ids = _create_record_ids(tp_df, 'TP')
        fp_ids = _create_record_ids(fp_df, 'FP')
        fn_ids = _create_record_ids(fn_df, 'FN')
        
        interval_data.append({
            'lower': lower,
            'upper': upper,
            'tp_ids': tp_ids,
            'fp_ids': fp_ids,
            'fn_ids': fn_ids
        })
    
    # Pass 2: Store raw counts for ALL distribution types using set operations
    histogram_data = {'x': [], 'tp_count': [], 'fp_count': [], 'fn_count': []}
    cumulative_data = {'x': [], 'tp_count': [], 'fp_count': [], 'fn_count': []}
    complementary_cumulative_data = {'x': [], 'tp_count': [], 'fp_count': [], 'fn_count': []}
    
    # First pass: compute histogram data
    for interval_info in interval_data:
        lower = interval_info['lower']
        upper = interval_info['upper']
        
        tp_count_hist = len(interval_info['tp_ids'])
        fp_count_hist = len(interval_info['fp_ids'])
        fn_count_hist = len(interval_info['fn_ids'])
        
        x_value_hist = np.sqrt(lower * upper)  # Geometric mean for log scale
        
        histogram_data['x'].append(x_value_hist)
        histogram_data['tp_count'].append(tp_count_hist)
        histogram_data['fp_count'].append(fp_count_hist)
        histogram_data['fn_count'].append(fn_count_hist)
    
    # Second pass: compute cumulative data with single forward pass
    tp_set_cum = set()
    fp_set_cum = set()
    fn_set_cum = set()
    for interval_info in interval_data:
        tp_set_cum |= interval_info['tp_ids']
        fp_set_cum |= interval_info['fp_ids']
        fn_set_cum |= interval_info['fn_ids']
        
        cumulative_data['x'].append(interval_info['upper'])
        cumulative_data['tp_count'].append(len(tp_set_cum))
        cumulative_data['fp_count'].append(len(fp_set_cum))
        cumulative_data['fn_count'].append(len(fn_set_cum))
    
    # Third pass: compute complementary cumulative data with single backward pass
    tp_set_comp = set()
    fp_set_comp = set()
    fn_set_comp = set()
    for i in range(len(interval_data) - 1, -1, -1):
        interval_info = interval_data[i]
        
        tp_set_comp |= interval_info['tp_ids']
        fp_set_comp |= interval_info['fp_ids']
        fn_set_comp |= interval_info['fn_ids']
        
        complementary_cumulative_data['x'].insert(0, interval_info['lower'])
        complementary_cumulative_data['tp_count'].insert(0, len(tp_set_comp))
        complementary_cumulative_data['fp_count'].insert(0, len(fp_set_comp))
        complementary_cumulative_data['fn_count'].insert(0, len(fn_set_comp))
    
    # Convert to numpy arrays and return
    key = (input_set_name, svtype)
    result = {
        DistributionType.HISTOGRAM: {
            'x': np.array(histogram_data['x']),
            'tp_count': np.array(histogram_data['tp_count']),
            'fp_count': np.array(histogram_data['fp_count']),
            'fn_count': np.array(histogram_data['fn_count'])
        },
        DistributionType.CUMULATIVE: {
            'x': np.array(cumulative_data['x']),
            'tp_count': np.array(cumulative_data['tp_count']),
            'fp_count': np.array(cumulative_data['fp_count']),
            'fn_count': np.array(cumulative_data['fn_count'])
        },
        DistributionType.COMPLEMENTARY_CUMULATIVE: {
            'x': np.array(complementary_cumulative_data['x']),
            'tp_count': np.array(complementary_cumulative_data['tp_count']),
            'fp_count': np.array(complementary_cumulative_data['fp_count']),
            'fn_count': np.array(complementary_cumulative_data['fn_count'])
        }
    }
    
    return key, result


class CNVPlotter:
    def __init__(self, data: dict, config: dict, input_name_mapping: dict):
        self.data = data
        self.config = config
        self.input_name_mapping = input_name_mapping
    
    def get_distribution_data(
        self,
        bounds: tuple[float, float],
        n_points: int = 50,
        svtypes: List[SVType] = [SVType.ALL, SVType.DEL, SVType.DUP],
        n_workers: Optional[int] = None,
    ):
        """
        Compute distribution data with raw TP/FP/FN counts across size ranges.
        
        Args:
            bounds: Tuple of (start, end) size range in bp
            n_points: Number of intervals to generate
            svtypes: List of SVType values to include
            n_workers: Number of worker processes (default: cpu_count() - 1)
        
        Returns:
            Dictionary mapping distribution_type -> {(input_set, svtype): data_dict}
            where data_dict contains 'x', 'tp_count', 'fp_count', 'fn_count' arrays
        """

        # Generate size intervals based on provided bounds and number of points
        start, end = bounds
        intervals = generate_size_intervals(start, end, n_points)
        
        # Initialize data structure for all distribution types
        data_by_distribution = {
            DistributionType.HISTOGRAM: {},
            DistributionType.CUMULATIVE: {},
            DistributionType.COMPLEMENTARY_CUMULATIVE: {}
        }
        
        # Prepare tasks for all (input_set, svtype) combinations
        tasks = [
            (input_set_name, svtype, analysis_data, intervals)
            for input_set_name, analysis_data in self.data.items()
            for svtype in svtypes
        ]
        
        # Set number of workers
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)
        
        # Limit workers to number of tasks
        n_workers = min(n_workers, len(tasks))
        
        print(f"Processing {len(tasks)} combinations using {n_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_process_input_sv_combination_worker, task): task 
                      for task in tasks}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                try:
                    key, result = future.result()
                    
                    # Merge result into data_by_distribution
                    for dist_type, data in result.items():
                        data_by_distribution[dist_type][key] = data
                    
                    completed += 1
                    if completed % max(1, len(tasks) // 10) == 0:
                        print(f"  Progress: {completed}/{len(tasks)} combinations completed")
                
                except Exception as e:
                    task_info = futures[future]
                    print(f"Error processing {task_info[0]}, {task_info[1]}: {e}")
                    raise
        
        print(f"✓ All {len(tasks)} combinations processed")

        return data_by_distribution

    def plot_statistical_distributions(
        self,
        metrics: List[Tuple[Callable[[int, int, int], float], str]],
        bounds: tuple[float, float],
        n_points: int = 100,
        svtypes: List[SVType] = [SVType.ALL, SVType.DEL, SVType.DUP],
        output_path: Optional[str | Path] = None,
        figsize: Tuple[int, int] = (12, 6),
        smoothing_sigma: float = 5.0,
        show_raw_points: bool = True,
    ):
        """
        Generate and plot statistical distributions of CNV performance metrics across size ranges.
        
        Creates three separate plots per metric (one for each distribution type: histogram, 
        cumulative, complementary_cumulative), each containing curves for all input_set/svtype 
        combinations.
        
        Args:
            metrics: List of (metric_function, metric_name) tuples where metric_function 
                     computes metric from (TP, FP, FN) counts
            bounds: Tuple of (start, end) size range in bp
            n_points: Number of intervals to generate
            svtypes: List of SVType values to plot
            output_path: Base path for output files (suffixed with metric and distribution type)
            figsize: Figure size tuple
            smoothing_sigma: Sigma for Gaussian smoothing (0 = no smoothing)
            show_raw_points: Whether to show raw data points beneath smoothed curves
        """

        time_0 = time.time()

        # Get distribution data for all types (raw counts)
        distribution_data = self.get_distribution_data(
            bounds=bounds,
            n_points=n_points,
            svtypes=svtypes
        )

        time_1 = time.time()
        
        # Set up color palette
        unique_input_sets = list(self.data.keys())
        cmap = colormaps["tab10"]
        color_map = {name: cmap(i % 10) for i, name in enumerate(unique_input_sets)}
        
        # Line styles for different svtypes
        linestyle_map = {
            SVType.ALL: '-',      # solid
            SVType.DEL: '--',     # dashed
            SVType.DUP: ':',      # dotted
        }
        
        # Plot each metric separately
        for metric_function, metric_name in metrics:
            # Plot each distribution type for this metric
            for dist_type, data_by_combination in distribution_data.items():
                fig, ax = plt.subplots(figsize=figsize)
                
                # Plot each combination of input_set and svtype
                for (input_set_name, svtype), data in data_by_combination.items():
                    if len(data['x']) == 0:
                        continue
                    
                    # Compute metric values from raw counts
                    y_values = np.array([
                        metric_function(tp, fp, fn)
                        for tp, fp, fn in zip(data['tp_count'], data['fp_count'], data['fn_count'])
                    ])
                    
                    # Sort by x-axis values
                    sort_idx = np.argsort(data['x'])
                    x_sorted = data['x'][sort_idx]
                    y_sorted = y_values[sort_idx]
                    
                    # Apply Gaussian smoothing if sigma > 0
                    if smoothing_sigma > 0 and len(y_sorted) > 1:
                        y_smoothed = gaussian_filter1d(y_sorted, sigma=smoothing_sigma)
                    else:
                        y_smoothed = y_sorted
                    
                    # Get color and line style
                    color = color_map.get(input_set_name, 'black')
                    linestyle = linestyle_map.get(svtype, '-')
                    
                    # Create label with display name
                    display_name = self.input_name_mapping.get(input_set_name, input_set_name)
                    label = f"{display_name} - {svtype.value if hasattr(svtype, 'value') else svtype}"
                    
                    # Adjust alpha and linewidth based on svtype prominence
                    if SVType.ALL in svtypes and len(svtypes) > 1:
                        if svtype == SVType.ALL:
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
                    if show_raw_points and (len(svtypes) == 1 or svtype == SVType.ALL):
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
                
                # Format plot
                ax.set_xlabel("CNV Size (bp)", fontsize=12)
                ax.set_ylabel(metric_name, fontsize=12)
                
                # Create title with metric name and distribution type
                dist_type_name = dist_type.value.replace('_', ' ').title() if hasattr(dist_type, 'value') else str(dist_type).replace('_', ' ').title()
                plot_title = f"{metric_name} by CNV Size - {dist_type_name}"
                ax.set_title(plot_title, fontsize=14, fontweight='bold')
                
                ax.legend(fontsize=9, title_fontsize=10, loc='best')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save or show
                if output_path:
                    output_base = Path(output_path)
                    output_dir = output_base.parent
                    output_dir.mkdir(parents=True, exist_ok=True)
                    stem = output_base.stem
                    suffix = output_base.suffix
                    
                    # Create filename with metric name and distribution type
                    metric_name_clean = metric_name.lower().replace(' ', '_')
                    dist_type_str = dist_type.value if hasattr(dist_type, 'value') else str(dist_type).split('.')[-1].lower()
                    plot_path = output_dir / f"{stem}_{metric_name_clean}_{dist_type_str}{suffix}"
                    
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    print(f"✓ Plot saved to: {plot_path}")
                else:
                    plt.show()
                
                plt.close()
        
        time_2 = time.time()
        print(f"Time to compute distribution data: {time_1 - time_0:.2f} seconds")
        print(f"Time to generate and save plots: {time_2 - time_1:.2f} seconds")

    
    def plot_venn_diagram(
        self,
        set_keys: List[str],
        bounds: Optional[Tuple[float, float]] = None,
        svtype: SVType = SVType.ALL,
        figsize: Tuple[int, int] = (10, 8),
        output_path: Optional[str | Path] = None,
    ):
        """
        Generate Venn diagram comparing TP/FP/FN sets for a specific SV type across all input sets.
        
        Args:
            svtype: SV type to filter by (e.g., 'DEL', 'DUP', or None for all)
            figsize: Figure size tuple
            output_path: Path to save the plot (if None, plot will be shown instead)
        """
        
        # Return if not exactly 3 set_keys (venn3 requires exactly 3 sets)
        if len(set_keys) != 3:
            print("Error: Venn diagram requires exactly 3 input sets.")
            return
        
        print(len(set_keys), "sets provided for Venn diagram:", set_keys)

        # Verify keys exist in data
        for key in set_keys:
            if key not in self.data:
                print(f"Error: Input set '{key}' not found in data.")
                return
        
        # Create output directory if saving
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Required columns for building benchmark IDs
        required_cols = ['truth_chrom', 'truth_start', 'truth_end', 'svtype']

        data = self.data.copy()  # Work with a copy to avoid modifying original data

        # Filter by size if bounds provided
        for key in set_keys:
            if bounds is not None:
                start, end = bounds
                data[key] = filter_by_size(data[key], lower_bound=int(start), upper_bound=int(end))
            

        # Extract TP tuples for each set
        tp_sets = {}
        for key in set_keys:
            df = data[key].get('TP', pd.DataFrame())

            if df.empty:
                print(f"Warning: No TP records found for input set '{key}'.")
                tp_sets[key] = set()
                continue
            
            # Filter by svtype if specified
            if svtype is not SVType.ALL and 'svtype' in df.columns:
                df = df[df['svtype'] == svtype.value].copy()
            
            # Check if required columns are present
            if all(col in df.columns for col in required_cols):
                tp_sets[key] = set(df[required_cols].itertuples(index=False, name=None))
            else:
                print(f"Warning: Required columns not found for input set '{key}'. Using row hashes instead.")
                tp_sets[key] = set(df.apply(lambda row: hash(tuple(row)), axis=1))
        
        # Calculate universal set (all benchmark IDs across all input sets)
        all_benchmark_ids = set().union(*tp_sets.values())
        total_unique_cnvs = len(all_benchmark_ids)
        
        # Extract FN (false negatives) from first set to get total truth records
        fn_df = data[set_keys[0]].get('FN', pd.DataFrame())
        fn_set = set()
        
        if not fn_df.empty:
            # Filter by svtype if specified
            if svtype is not SVType.ALL and 'svtype' in fn_df.columns:
                fn_df = fn_df[fn_df['svtype'] == svtype.value].copy()
            
            # Extract FN IDs
            if all(col in fn_df.columns for col in required_cols):
                fn_set = set(fn_df[required_cols].itertuples(index=False, name=None))
            else:
                fn_set = set(fn_df.apply(lambda row: hash(tuple(row)), axis=1))
        
        # Total truth records = all detected (TP from any method) + not detected (FN)
        all_truth_ids = all_benchmark_ids.union(fn_set)
        total_truth_cnvs = len(all_truth_ids)
        
        # Apply name mapping and add counts to labels
        display_names_with_counts = []
        for input_set_key in set_keys:
            count = len(tp_sets[input_set_key])
            pct_total = (count / total_unique_cnvs) * 100 if total_unique_cnvs > 0 else 0
            
            display_name = self.input_name_mapping.get(input_set_key, input_set_key)
            label = f"{display_name}\n(n={count}, {pct_total:.1f}%)"
            display_names_with_counts.append(label)
        
        # Create Venn diagram
        fig, ax = plt.subplots(figsize=figsize)
        
        display_names_tuple: Tuple[str, str, str] = (
            display_names_with_counts[0], 
            display_names_with_counts[1], 
            display_names_with_counts[2]
        )
        venn_obj = venn3(
            subsets=tuple(tp_sets.values()),
            set_labels=display_names_tuple,
            ax=ax
        )
        venn_circles_obj = venn3_circles(subsets=tuple(tp_sets.values()), ax=ax)
        
        # Customize appearance
        for circle in venn_circles_obj:
            circle.set_linewidth(2)
            circle.set_linestyle('--')
        
        # Generate title
        svtype_str = f" ({svtype.value})" if svtype != SVType.ALL else ""
        title = f"Benchmark CNV Recall Overlap{svtype_str}\n({total_unique_cnvs} detected of {total_truth_cnvs} total truth records)"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Calculate statistics for text box
        total_detected_by_at_least_one = total_unique_cnvs  # All IDs in all_benchmark_ids are detected by at least one method
        recall_rate = (total_unique_cnvs / total_truth_cnvs * 100) if total_truth_cnvs > 0 else 0
        
        # Add summary statistics text box
        stats_text = (
            f"Total truth CNVs: {total_truth_cnvs} | "
            f"Detected by ≥1 method: {total_detected_by_at_least_one} ({recall_rate:.1f}%) | "
            f"Not detected: {total_truth_cnvs - total_unique_cnvs}"
        )
        ax.text(
            0.5, -0.15, stats_text, 
            ha='center', 
            transform=ax.transAxes,
            fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Venn diagram saved to: {output_path}")
        else:
            plt.show()
        plt.close()

        # === Print detailed statistics ===
        print(f"\n{'='*60}")
        print(f"Venn Diagram Detection Statistics")
        print(f"{'='*60}")
        print(f"Total truth benchmark CNVs (TP + FN): {total_truth_cnvs}")
        print(f"Detected by at least one method: {total_detected_by_at_least_one} ({recall_rate:.1f}% of truth)")
        print(f"Not detected by any method (FN): {total_truth_cnvs - total_unique_cnvs}")
        
        print(f"\nDetection by individual methods:")
        for input_set_key in set_keys:
            count = len(tp_sets[input_set_key])
            display_name = self.input_name_mapping.get(input_set_key, input_set_key)
            pct_truth = (count / total_truth_cnvs) * 100 if total_truth_cnvs > 0 else 0
            pct_detected = (count / total_unique_cnvs) * 100 if total_unique_cnvs > 0 else 0
            print(f"  {display_name}: {count} ({pct_truth:.1f}% of truth, {pct_detected:.1f}% of detected)")
        
        print(f"\nDetailed Overlap Counts:")
        subsets = venn_obj.get_label_by_id
        combinations = ['100', '010', '001', '110', '101', '011', '111']
        
        for comb in combinations:
            label = subsets(comb)
            if label is None:
                count = 0
            else:
                count = int(label.get_text())
            detected = (count / total_unique_cnvs) * 100 if total_unique_cnvs > 0 else 0
            pct_truth = (count / total_truth_cnvs) * 100 if total_truth_cnvs > 0 else 0
            
            # Calculate % of each method involved in this combination
            involved_indices = [i for i, bit in enumerate(comb) if bit == '1']
            pct_each_method = []
            for i in involved_indices:
                method_count = len(tp_sets[set_keys[i]])
                pct_method = (count / method_count) * 100 if method_count > 0 else 0
                method_name = self.input_name_mapping.get(set_keys[i], set_keys[i])
                pct_each_method.append(f"{method_name}: {pct_method:.1f}%")
            pct_each_method_str = "| ".join(pct_each_method)
            
            print(f"  Combination {comb}: {count} ({detected:.1f}% of detected, {pct_truth:.1f}% of truth)")
            print(f"    - {pct_each_method_str}")


    def plot_size_distribution(
        self,
        set_keys: List[str],
        svtype: SVType = SVType.ALL,
        figsize: Tuple[int, int] = (12, 6),
        output_dir: Optional[str | Path] = None,
        include_benchmark: bool = True,
    ):
        """
        Generate size distribution plots (histogram and KDE) for CNVs.
        
        Args:
            set_keys: List of input set keys to plot
            svtype: SV type to filter by (DEL, DUP, or ALL)
            figsize: Figure size tuple
            output_dir: Directory to save plots (if None, plots will be shown)
            include_benchmark: Whether to include benchmark truth set in plots
        """

        # Prepare output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        svtype_str = f" ({svtype.value})" if svtype != SVType.ALL else ""
        
        # Collect all data into a single DataFrame
        all_data = []
        
        # Collect prediction data (TP + FP)
        for set_key in set_keys:
            for classification in ['TP', 'FP']:
                df = self.data[set_key].get(classification)
                if df is None or df.empty or 'pred_size' not in df.columns:
                    continue
                
                # Apply svtype filter
                if svtype != SVType.ALL and 'svtype' in df.columns:
                    df = df[df['svtype'] == svtype.value].copy()
                
                # Extract sizes and add metadata
                sizes = df['pred_size'].dropna()
                sizes = sizes[sizes > 0]
                
                if len(sizes) > 0:
                    display_name = self.input_name_mapping.get(set_key, set_key)
                    temp_df = pd.DataFrame({
                        'size': sizes,
                        'source': display_name,
                        'type': 'prediction'
                    })
                    all_data.append(temp_df)
        
        # Collect benchmark data (TP + FN) if requested
        if include_benchmark and set_keys:
            for classification in ['TP', 'FN']:
                df = self.data[set_keys[0]].get(classification)
                if df is None or df.empty or 'truth_size' not in df.columns:
                    continue
                
                # Apply svtype filter
                if svtype != SVType.ALL and 'svtype' in df.columns:
                    df = df[df['svtype'] == svtype.value].copy()
                
                # Extract sizes and add metadata
                sizes = df['truth_size'].dropna()
                sizes = sizes[sizes > 0]
                
                if len(sizes) > 0:
                    temp_df = pd.DataFrame({
                        'size': sizes,
                        'source': 'Benchmark (Truth)',
                        'type': 'benchmark'
                    })
                    all_data.append(temp_df)
        
        # Combine all data
        if not all_data:
            print("Warning: No data available for plotting.")
            return
        
        plot_df = pd.concat(all_data, ignore_index=True)

        # Get min and max sizes for setting x-axis limits
        min_size = plot_df['size'].min()
        max_size = plot_df['size'].max()
        
        # ==================== PLOT 1: HISTOGRAM ====================
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.histplot(
            data=plot_df,
            x='size',
            hue='source',
            log_scale=True,
            element='step',
            stat='density',
            common_norm=False,
            linewidth=2,
            legend=True,
            ax=ax
        )
        
        ax.set_xlabel("Size (bp)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xlim(min_size * 0.9, max_size * 1.1)
        title_suffix = " (with Benchmark)" if include_benchmark else ""
        ax.set_title(f"CNV Size Distribution - Histogram{svtype_str}{title_suffix}", 
                    fontsize=14, fontweight='bold')
        
        # Customize legend
        legend = ax.get_legend()
        if legend:
            legend.set_title('Source')
            plt.setp(legend.get_texts(), fontsize=10)
            plt.setp(legend.get_title(), fontsize=10)


        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_dir:
            histogram_path = output_dir / f"size_distribution_histogram{svtype_str.replace(' ', '_')}.png"
            plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
            print(f"✓ Histogram saved to: {histogram_path}")
        else:
            plt.show()
        plt.close()
        
        # ==================== PLOT 2: KDE ====================
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.kdeplot(
            data=plot_df,
            x='size',
            hue='source',
            log_scale=True,
            common_norm=False,
            fill=True,
            alpha=0.4,
            linewidth=2.5,
            legend=True,
            ax=ax
        )
        
        ax.set_xlabel("Size (bp)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xlim(min_size * 0.9, max_size * 1.1)
        ax.set_title(f"CNV Size Distribution - KDE{svtype_str}{title_suffix}", 
                    fontsize=14, fontweight='bold')
        
        # Customize legend
        legend = ax.get_legend()
        if legend:
            legend.set_title('Source')
            plt.setp(legend.get_texts(), fontsize=10)
            plt.setp(legend.get_title(), fontsize=10)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_dir:
            kde_path = output_dir / f"size_distribution_kde{svtype_str.replace(' ', '_')}.png"
            plt.savefig(kde_path, dpi=300, bbox_inches='tight')
            print(f"✓ KDE saved to: {kde_path}")
        else:
            plt.show()
        plt.close()
        
        print(f"✓ Size distribution plots completed!")




            