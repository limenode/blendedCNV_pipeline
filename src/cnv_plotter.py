from collections import defaultdict
from typing import Callable, Tuple, Optional, List, Dict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib_venn import venn3, venn3_circles
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns
from matplotlib.patches import Rectangle
import os

from load_analysis_data import filter_by_size
from utils import generate_size_intervals, DistributionType, SVType

# Module-level helper function for multiprocessing
def _create_record_ids(df: pd.DataFrame, classification: str, svtype: SVType = SVType.ALL) -> set:
    """
    Create unique identifiers for records based on classification type.
    
    Args:
        df: DataFrame with records
        classification: One of 'TP', 'FP', or 'FN'
        svtype: The structural variant type to filter by
    
    Returns:
        Set of tuples uniquely identifying each record
    """
    if df.empty:
        return set()
    
    if svtype != SVType.ALL and 'svtype' in df.columns:
        df = df[df['svtype'] == svtype.value].copy()
    
    if classification in ['FP']:
        # Use predicted coordinates for FP + sample
        required_cols = ['pred_chrom', 'pred_start', 'pred_end', 'svtype', 'sample']
        if all(col in df.columns for col in required_cols):
            return set(df[required_cols].itertuples(index=False, name=None))
    elif classification in ['TP', 'FN']:
        # Use truth coordinates for TP and FN + sample
        required_cols = ['truth_chrom', 'truth_start', 'truth_end', 'svtype', 'sample']
        if all(col in df.columns for col in required_cols):
            return set(df[required_cols].itertuples(index=False, name=None))
    
    # Fallback: use row hashes if required columns not available
    return set(df.apply(lambda row: hash(tuple(row)), axis=1))


def _process_input_sv_combination_worker(args):
    """
    Worker function to process a single (input_set_name, svtype) combination.
    Designed to be called from multiprocessing pool.
    
    Args:
        args: Tuple of (input_set_name, svtype, analysis_data, intervals, undiscoverable_fns)
    
    Returns:
        Tuple of (key, result_dict) where key is (input_set_name, svtype)
        and result_dict contains data for all three distribution types
    """
    input_set_name, svtype, analysis_data, intervals, undiscoverable_fns = args
    
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
        
        # Exclude undiscoverable FNs if provided
        if undiscoverable_fns:
            fn_ids -= undiscoverable_fns
        
        interval_data.append({
            'lower': lower,
            'upper': upper,
            'tp_ids': tp_ids,
            'fp_ids': fp_ids,
            'fn_ids': fn_ids
        })
    
    # Pass 2: Store raw counts for ALL distribution types using set operations
    density_data = {'x': [], 'tp_count': [], 'fp_count': [], 'fn_count': []}
    cumulative_data = {'x': [], 'tp_count': [], 'fp_count': [], 'fn_count': []}
    complementary_cumulative_data = {'x': [], 'tp_count': [], 'fp_count': [], 'fn_count': []}
    
    # First pass: compute density distribution data
    for interval_info in interval_data:
        lower = interval_info['lower']
        upper = interval_info['upper']
        
        tp_count_density = len(interval_info['tp_ids'])
        fp_count_density = len(interval_info['fp_ids'])
        fn_count_density = len(interval_info['fn_ids'])
        
        x_value_density = np.sqrt(lower * upper)  # Geometric mean for log scale
        
        density_data['x'].append(x_value_density)
        density_data['tp_count'].append(tp_count_density)
        density_data['fp_count'].append(fp_count_density)
        density_data['fn_count'].append(fn_count_density)
    
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
        DistributionType.DENSITY: {
            'x': np.array(density_data['x']),
            'tp_count': np.array(density_data['tp_count']),
            'fp_count': np.array(density_data['fp_count']),
            'fn_count': np.array(density_data['fn_count'])
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


def _plot_single_metric_distribution_worker(
    metric_function: Callable[[int, int, int], float], 
    metric_name: str, dist_type: DistributionType, 
    data_by_combination: dict, 
    color_map: dict, 
    linestyle_map: dict, 
    input_name_mapping: dict, 
    input_set_order: List[str],
    svtypes: list, 
    figsize: tuple, 
    smoothing_sigma: float, 
    show_raw_points: bool, 
    output_dir: str | Path
):
    """
    Worker function to create a single plot for a metric/distribution combination.
    
    Returns:
        Tuple of (metric_name, dist_type, success_flag, output_path)
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each combination in explicit order so legend order is deterministic.
        for input_set_name in input_set_order:
            for svtype in svtypes:
                data = data_by_combination.get((input_set_name, svtype))
                if data is None or len(data['x']) == 0:
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
                display_name = input_name_mapping.get(input_set_name, input_set_name)
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
        
        # Save the plot
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with metric name and distribution type
        metric_name_clean = metric_name.lower().replace(' ', '_').replace('/', '_')
        dist_type_str = dist_type.value if hasattr(dist_type, 'value') else str(dist_type).split('.')[-1].lower()
        plot_path = output_dir / f"{metric_name_clean}_{dist_type_str}.png"
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return (metric_name, dist_type, True, str(plot_path))
    
    except Exception as e:
        plt.close('all')
        return (metric_name, dist_type, False, str(e))


def identify_undiscoverable_cnvs(
        data: dict, 
) -> set:

    input_sets = data.get('input_sets', {})

    # Retrieve all FN dataframes, filter, and add to list for concatenation
    fn_id_sets = []

    for input_set_name, analysis_data in input_sets.items():

        if 'FN' not in analysis_data:
            continue
        
        fn_df = analysis_data['FN'].copy()

        # Get IDs for all FNs and add to set
        fn_id_set = _create_record_ids(fn_df, 'FN')

        fn_id_sets.append(fn_id_set)
    
    print(f"Collected FN ID sets from {len(fn_id_sets)} input sets for undiscoverable CNV analysis")
    
    # Find intersection of all FN ID sets to identify undiscoverable CNVs
    if fn_id_sets:
        undiscoverable_cnvs = set.intersection(*fn_id_sets)
        print(f"Identified {len(undiscoverable_cnvs)} undiscoverable CNVs present as FNs in all input sets")
    else:
        undiscoverable_cnvs = set()
        print("No FN data found across input sets to identify undiscoverable CNVs")

    return undiscoverable_cnvs

# Compute summary size statistics by source.
def _compute_series_stats(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return pd.Series({
        'count': int(series.count()),
        'min': float(series.min()),
        'q1': float(q1),
        'median': float(series.median()),
        'q3': float(q3),
        'iqr': float(q3 - q1),
        'mean': float(series.mean()),
        'stdev': float(series.std(ddof=1)) if series.count() > 1 else np.nan,
        'max': float(series.max()),
    })

def _filter_svtype(df: pd.DataFrame, svtype: SVType) -> pd.DataFrame:
    if df.empty:
        return df
    if svtype != SVType.ALL and 'svtype' in df.columns:
        return df[df['svtype'] == svtype.value].copy()
    return df

class CNVPlotter:
    def __init__(self, data: dict, config: dict, input_name_mapping: dict):
        self.data = data
        self.config = config
        self.input_name_mapping = input_name_mapping

    def _build_tp_and_truth_sets(
        self,
        set_keys: List[str],
        data: Dict[str, dict],
        svtype: SVType,
    ) -> Tuple[Dict[str, set], set, int, Dict[str, set], set, int, Dict[str, int]]:
        """
        Build shared TP and truth (TP U FN) sets used by overlap visualizations.

        Returns:
            Tuple of:
                tp_sets,
                detected_ids,
                total_detected,
                truth_ids_by_method,
                all_truth_ids,
                total_truth_cnvs,
                truth_set_sizes
        """
        tp_sets: Dict[str, set] = {}
        for key in set_keys:
            tp_df = data[key].get('TP', pd.DataFrame())
            tp_sets[key] = _create_record_ids(tp_df, 'TP', svtype=svtype)

        detected_ids = set().union(*tp_sets.values()) if tp_sets else set()
        total_detected = len(detected_ids)

        truth_ids_by_method: Dict[str, set] = {}
        for key in set_keys:
            fn_df = data[key].get('FN', pd.DataFrame())
            fn_ids = _create_record_ids(fn_df, 'FN', svtype=svtype)
            truth_ids_by_method[key] = tp_sets[key].union(fn_ids)

        all_truth_ids = set().union(*truth_ids_by_method.values()) if truth_ids_by_method else set()
        total_truth_cnvs = len(all_truth_ids)
        truth_set_sizes = {k: len(v) for k, v in truth_ids_by_method.items()}

        return (
            tp_sets,
            detected_ids,
            total_detected,
            truth_ids_by_method,
            all_truth_ids,
            total_truth_cnvs,
            truth_set_sizes,
        )
    
    def get_distribution_data(
        self,
        unique_input_sets: set,
        bounds: tuple[float, float],
        n_points: int = 50,
        svtypes: List[SVType] = [SVType.ALL, SVType.DEL, SVType.DUP],
        n_workers: Optional[int] = None,
    ) -> Dict[DistributionType, Dict[Tuple[str, SVType], dict]]:
        """
        Compute distribution data with raw TP/FP/FN counts across size ranges.
        
        Args:
            unique_input_sets: Set of input set keys to include in the distribution data
            bounds: Tuple of (start, end) size range in bp
            n_points: Number of intervals to generate
            svtypes: List of SVType values to include
            n_workers: Number of worker processes (default: cpu_count() - 1)
            exclude_undiscoverable_fn: If True, exclude FNs present in all datasets (undiscoverable)
        
        Returns:
            Dictionary mapping distribution_type -> {(input_set, svtype): data_dict}
            where data_dict contains 'x', 'tp_count', 'fp_count', 'fn_count' arrays
        """

        # Generate size intervals based on provided bounds and number of points
        start, end = bounds
        intervals = generate_size_intervals(start, end, n_points)
        
        # Initialize data structure for all distribution types
        data_by_distribution = {
            DistributionType.DENSITY: {},
            DistributionType.CUMULATIVE: {},
            DistributionType.COMPLEMENTARY_CUMULATIVE: {}
        }
        
        undiscoverable_cnvs = identify_undiscoverable_cnvs(self.data)
        
        print(len(undiscoverable_cnvs))

        # Prepare tasks for all (input_set, svtype) combinations
        # Only iterate over input_sets
        input_sets = self.data.get('input_sets', {})
        tasks = [
            (input_set_name, svtype, analysis_data, intervals, undiscoverable_cnvs)
            for input_set_name, analysis_data in input_sets.items() if input_set_name in unique_input_sets
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

    def _generate_cumulative_stats_summary_table(self, distribution_data, metrics, cumulative_stats_output_path):
        # Build and optionally save summary metrics from the final cumulative interval.
        cumulative_summary_rows = []
        cumulative_data = distribution_data.get(DistributionType.CUMULATIVE, {})
        for (input_set_name, svtype), data in cumulative_data.items():
            if len(data.get('x', [])) == 0:
                continue

            tp_final = int(data['tp_count'][-1])
            fp_final = int(data['fp_count'][-1])
            fn_final = int(data['fn_count'][-1])

            row = {
                'input_set': input_set_name,
                'input_set_display': self.input_name_mapping.get(input_set_name, input_set_name),
                'svtype': svtype.value if hasattr(svtype, 'value') else str(svtype),
                'analysis_window_end': float(data['x'][-1]),
                'tp_count': tp_final,
                'fp_count': fp_final,
                'fn_count': fn_final,
            }

            for metric_function, metric_name in metrics:
                row[metric_name] = metric_function(tp_final, fp_final, fn_final)

            cumulative_summary_rows.append(row)

        cumulative_summary_df = pd.DataFrame(cumulative_summary_rows)    

        if cumulative_stats_output_path is not None:
            resolved_cumulative_stats_output = Path(cumulative_stats_output_path)
            resolved_cumulative_stats_output.parent.mkdir(parents=True, exist_ok=True)
            file_ext = resolved_cumulative_stats_output.suffix.lower()
            if file_ext in ['.tsv', '.txt']:
                cumulative_summary_df.to_csv(resolved_cumulative_stats_output, sep='\t', index=False)
            else:
                cumulative_summary_df.to_csv(resolved_cumulative_stats_output, index=False)
            print(f"✓ Final cumulative summary table saved to: {resolved_cumulative_stats_output}")
        else:
            print("Final cumulative summary table (not saved; provide cumulative_stats_output_path or output_path):")
            if cumulative_summary_df.empty:
                print("No cumulative data available.")
            else:
                print(cumulative_summary_df.to_string(index=False))

    def plot_statistical_distributions(
        self,
        input_sets_to_plot: Dict[str, List[str]],
        metrics: List[Tuple[Callable[[int, int, int], float], str]],
        bounds: tuple[float, float],
        n_points: int = 100,
        svtypes: List[SVType] = [SVType.ALL, SVType.DEL, SVType.DUP],
        output_dir: Optional[str | Path] = None,
        cumulative_stats_output_path: Optional[str | Path] = None,
        figsize: Tuple[int, int] = (12, 6),
        smoothing_sigma: float = 5.0,
        show_raw_points: bool = True,
    ):
        """
        Generate and plot statistical distributions of CNV performance metrics across size ranges.
        
        Creates three separate plots per metric (one for each distribution type: density, 
        cumulative, complementary_cumulative), each containing curves for all input_set/svtype 
        combinations.
        
        Args:
            input_sets_to_plot: Dict mapping plot group name -> list of input set keys
                                (e.g., {'Intersections': ['30x', '6x'], 'Unions': ['30x', 'cnvpytor']})
            metrics: List of (metric_function, metric_name) tuples where metric_function 
                     computes metric from (TP, FP, FN) counts
            bounds: Tuple of (start, end) size range in bp
            n_points: Number of intervals to generate
            svtypes: List of SVType values to plot
            output_path: Base path for output files (suffixed with metric and distribution type)
            cumulative_stats_output_path: Path to save final cumulative metric summary table (csv/tsv)
            figsize: Figure size tuple
            smoothing_sigma: Sigma for Gaussian smoothing (0 = no smoothing)
            show_raw_points: Whether to show raw data points beneath smoothed curves
        """

        # Get unique items from input_sets_to_plot for validation
        unique_input_sets = set()
        for input_set_list in input_sets_to_plot.values():
            unique_input_sets.update(input_set_list)

        # Get distribution data for all types (raw counts)
        distribution_data = self.get_distribution_data(
            unique_input_sets=unique_input_sets,
            bounds=bounds,
            n_points=n_points,
            svtypes=svtypes,
        )

        # Generate cumulative stats summary table from the final cumulative interval and save to file
        self._generate_cumulative_stats_summary_table(
            distribution_data=distribution_data,
            metrics=metrics,
            cumulative_stats_output_path=cumulative_stats_output_path
        )
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            exit(1)
        
        # Line styles for different svtypes
        linestyle_map = {
            SVType.ALL: '-',      # solid
            SVType.DEL: '--',     # dashed
            SVType.DUP: ':',      # dotted
        }

        cmap = colormaps['tab10']
        
        # Prepare plotting tasks for parallel execution
        plotting_tasks = []
        for plot_group_name, plot_group_input_sets in input_sets_to_plot.items():
            plot_group_input_set = set(plot_group_input_sets)

            # Keep output files separated by plot group name (create directory if it doesn't exist).
            plot_group_name_clean = str(plot_group_name).lower().replace(' ', '_').replace('/', '_')
            output_subdir = output_dir / plot_group_name_clean

            for metric_function, metric_name in metrics:
                for dist_type, data_by_combination in distribution_data.items():
                    # Filter data_by_combination to only include input sets relevant to this plot group.
                    filtered_data_by_combination = {
                        key: data
                        for key, data in data_by_combination.items()
                        if key[0] in plot_group_input_set
                    }

                    color_map = {input_set: cmap(i) for i, input_set in enumerate(plot_group_input_sets)}

                    task_args = (
                        metric_function,
                        metric_name,
                        dist_type,
                        filtered_data_by_combination,
                        color_map,
                        linestyle_map,
                        self.input_name_mapping,
                        plot_group_input_sets,
                        svtypes,
                        figsize,
                        smoothing_sigma,
                        show_raw_points,
                        output_subdir,
                    )
                    plotting_tasks.append(task_args)
        
        # Execute plotting tasks in parallel
        num_tasks = len(plotting_tasks)
        print(f"\nGenerating {num_tasks} plots in parallel ({len(metrics)} metrics x {len(distribution_data)} distribution types)...")
        
        successful_plots = []
        failed_plots = []
        
        with ProcessPoolExecutor(max_workers=min(num_tasks, cpu_count())) as executor:
            futures = [executor.submit(_plot_single_metric_distribution_worker, *task) 
                      for task in plotting_tasks]
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    metric_name, dist_type, success, result = future.result()
                    if success:
                        successful_plots.append((metric_name, dist_type, result))
                        print(f"✓ [{completed}/{num_tasks}] Plot saved: {Path(result).name}")
                    else:
                        failed_plots.append((metric_name, dist_type, result))
                        print(f"✗ [{completed}/{num_tasks}] Failed: {metric_name} - {dist_type}: {result}")
                except Exception as e:
                    completed += 1
                    print(f"✗ [{completed}/{num_tasks}] Error: {e}")
                    import traceback
                    traceback.print_exc()
    
    def plot_count_venn_diagram(
        self,
        config: dict,
        input_set_key: str,
        bounds: Optional[Tuple[float, float]] = None,
        svtype: SVType = SVType.ALL,
        figsize: Tuple[int, int] = (10, 8),
        output_path: Optional[str | Path] = None,
    ):
        """
        Plot caller overlap counts from TP+FP records in a 1-of-3 consensus input set.

        This function expects exactly three caller names under:
        config['input'][input_set_key].keys()
        and uses TP/FP rows from:
        self.data['input_sets'][f"{input_set_key}_consensus_1of3_intersections"]
        """
        input_cfg = config.get('input', {}).get(input_set_key)
        if input_cfg is None:
            print(f"Error: input_set_key '{input_set_key}' not found under config['input'].")
            return

        caller_names = list(input_cfg.keys())
        if len(caller_names) != 3:
            print(
                f"Error: Expected exactly 3 callers in config['input']['{input_set_key}'], "
                f"found {len(caller_names)}: {caller_names}"
            )
            return

        intersection_key = f"{input_set_key.replace(' ', '_')}_consensus_1of3_intersections"
        intersection_data = self.data.get('input_sets', {}).get(intersection_key)
        if intersection_data is None:
            print(f"Error: Input set '{intersection_key}' not found in self.data['input_sets'].")
            return

        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

        if bounds is not None:
            start, end = bounds
            intersection_data = filter_by_size(
                intersection_data,
                lower_bound=int(start),
                upper_bound=int(end),
            )

        tp_df = intersection_data.get('TP', pd.DataFrame()).copy()
        fp_df = intersection_data.get('FP', pd.DataFrame()).copy()

        if svtype != SVType.ALL:
            if not tp_df.empty and 'svtype' in tp_df.columns:
                tp_df = tp_df[tp_df['svtype'] == svtype.value].copy()
            if not fp_df.empty and 'svtype' in fp_df.columns:
                fp_df = fp_df[fp_df['svtype'] == svtype.value].copy()

        source_parts: List[pd.Series] = []
        for df in [tp_df, fp_df]:
            if df.empty:
                continue
            if 'sources' not in df.columns:
                print("Warning: Missing 'sources' column in one TP/FP dataframe; skipping it.")
                continue
            source_parts.append(df['sources'].dropna().astype(str).str.strip())

        if not source_parts:
            print("Warning: No TP/FP 'sources' data available to plot.")
            return

        all_sources = pd.concat(source_parts, ignore_index=True)
        if all_sources.empty:
            print("Warning: No valid TP/FP source combinations found to plot.")
            return

        source_dummies = all_sources.str.get_dummies(sep='|')
        total_source_rows = len(source_dummies)

        unexpected_callers = [col for col in source_dummies.columns if col not in caller_names]
        skipped_invalid_sources = 0
        if unexpected_callers:
            invalid_mask = source_dummies[unexpected_callers].sum(axis=1) > 0
            skipped_invalid_sources = int(invalid_mask.sum())
            source_dummies = source_dummies.loc[~invalid_mask].copy()
            print(
                "Warning: Found unexpected callers in 'sources' and skipped affected rows: "
                f"{unexpected_callers}"
            )

        source_dummies = source_dummies.reindex(columns=caller_names, fill_value=0).clip(upper=1)
        source_dummies = source_dummies[source_dummies.sum(axis=1) > 0].copy()

        if source_dummies.empty:
            print("Warning: No valid TP/FP source combinations found to plot.")
            return

        a, b, c = caller_names

        overlap_counts = {
            '100': int(((source_dummies[a] == 1) & (source_dummies[b] == 0) & (source_dummies[c] == 0)).sum()),
            '010': int(((source_dummies[a] == 0) & (source_dummies[b] == 1) & (source_dummies[c] == 0)).sum()),
            '001': int(((source_dummies[a] == 0) & (source_dummies[b] == 0) & (source_dummies[c] == 1)).sum()),
            '110': int(((source_dummies[a] == 1) & (source_dummies[b] == 1) & (source_dummies[c] == 0)).sum()),
            '101': int(((source_dummies[a] == 1) & (source_dummies[b] == 0) & (source_dummies[c] == 1)).sum()),
            '011': int(((source_dummies[a] == 0) & (source_dummies[b] == 1) & (source_dummies[c] == 1)).sum()),
            '111': int(((source_dummies[a] == 1) & (source_dummies[b] == 1) & (source_dummies[c] == 1)).sum()),
        }

        combo_series = source_dummies.apply(
            lambda row: "|".join(
                [caller for caller in caller_names if int(row[caller]) == 1]
            ),
            axis=1,
        )
        combo_counts: Dict[str, int] = {
            str(k): int(v)
            for k, v in combo_series.value_counts().items()
        }

        total_calls = sum(overlap_counts.values())
        if total_calls == 0:
            print("Warning: All overlap counts are zero after filtering.")
            return

        caller_totals = {
            a: overlap_counts['100'] + overlap_counts['110'] + overlap_counts['101'] + overlap_counts['111'],
            b: overlap_counts['010'] + overlap_counts['110'] + overlap_counts['011'] + overlap_counts['111'],
            c: overlap_counts['001'] + overlap_counts['101'] + overlap_counts['011'] + overlap_counts['111'],
        }

        caller_labels = [
            f"{caller}\n(n={caller_totals[caller]}, {(caller_totals[caller] / total_calls) * 100:.1f}%)"
            for caller in [a, b, c]
        ]

        venn_subsets = (
            overlap_counts['100'],
            overlap_counts['010'],
            overlap_counts['110'],
            overlap_counts['001'],
            overlap_counts['101'],
            overlap_counts['011'],
            overlap_counts['111'],
        )
        caller_labels_tuple: Tuple[str, str, str] = (
            caller_labels[0],
            caller_labels[1],
            caller_labels[2],
        )

        fig, ax = plt.subplots(figsize=figsize)
        venn_obj = venn3(subsets=venn_subsets, set_labels=caller_labels_tuple, ax=ax)
        venn_circles_obj = venn3_circles(subsets=venn_subsets, ax=ax)

        for region_id, count in overlap_counts.items():
            region_label = venn_obj.get_label_by_id(region_id)
            if region_label is not None:
                region_label.set_text(str(count))

        for circle in venn_circles_obj:
            circle.set_linewidth(2)
            circle.set_linestyle('--')

        svtype_str = f" ({svtype.value})" if svtype != SVType.ALL else ""
        title = f"Caller Source Overlap (TP+FP){svtype_str}\n{input_set_key} | n={total_calls}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        skipped_text = ""
        if skipped_invalid_sources > 0:
            skipped_text = f" | Skipped invalid source rows: {skipped_invalid_sources}/{total_source_rows}"
        ax.text(
            0.5,
            -0.12,
            f"Total counted TP+FP records: {total_calls}{skipped_text}",
            ha='center',
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
        )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Venn diagram saved to: {output_path}")
        else:
            plt.show()
        plt.close()

        print(f"\n{'=' * 60}")
        print("Count Venn Diagram Statistics (TP + FP Sources)")
        print(f"{'=' * 60}")
        print(f"Input set: {input_set_key}")
        print(f"Consensus source set: {intersection_key}")
        print(f"Callers from config: {caller_names}")
        print(f"Total counted TP+FP records: {total_calls}")
        if skipped_invalid_sources > 0:
            print(f"Skipped invalid source rows: {skipped_invalid_sources}/{total_source_rows}")

        print("\nPer-region counts:")
        region_order = ['100', '010', '001', '110', '101', '011', '111']
        for region_id in region_order:
            count = overlap_counts[region_id]
            pct = (count / total_calls) * 100 if total_calls > 0 else 0.0
            print(f"  {region_id}: {count} ({pct:.1f}%)")

        print("\nRaw source combination counts:")
        for combo in sorted(combo_counts.keys(), key=lambda x: (x.count('|'), x)):
            count = combo_counts[combo]
            pct = (count / total_calls) * 100 if total_calls > 0 else 0.0
            print(f"  {combo}: {count} ({pct:.1f}%)")


    def plot_recall_venn_diagram(
        self,
        set_keys: List[str],
        bounds: Optional[Tuple[float, float]] = None,
        svtype: SVType = SVType.ALL,
        figsize: Tuple[int, int] = (10, 8),
        output_path: Optional[str | Path] = None,
        show_region_table: bool = False,
    ):
        """
        Generate Venn diagram comparing TP/FP/FN sets for a specific SV type across all input sets.
        
        Args:
            svtype: SV type to filter by (e.g., 'DEL', 'DUP', or None for all)
            figsize: Figure size tuple
            output_path: Path to save the plot (if None, plot will be shown instead)
            show_region_table: If True, add a side panel listing all 7 region counts
        """
        
        # Return if not exactly 3 set_keys (venn3 requires exactly 3 sets)
        if len(set_keys) != 3:
            print("Error: Venn diagram requires exactly 3 input sets.")
            return
        
        print(len(set_keys), "sets provided for Venn diagram:", set_keys)

        # Verify keys exist in data
        for key in set_keys:
            if key not in self.data.get('input_sets', {}):
                print(f"Error: Input set '{key}' not found in data.")
                return
        
        # Create output directory if saving
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

        # Work with copy of input_sets only
        data = self.data.get('input_sets', {}).copy()

        # Filter by size if bounds provided
        for key in set_keys:
            if bounds is not None:
                start, end = bounds
                data[key] = filter_by_size(data[key], lower_bound=int(start), upper_bound=int(end))
        
        # Build shared TP/truth sets
        (
            tp_sets,
            all_benchmark_ids,
            total_unique_cnvs,
            truth_ids_by_method,
            all_truth_ids,
            total_truth_cnvs,
            truth_set_sizes,
        ) = self._build_tp_and_truth_sets(set_keys=set_keys, data=data, svtype=svtype)

        print(sum(len(s) for s in tp_sets.values()), "total TP records across all sets after using _create_record_ids with svtype filtering.")

        # Compute exact Venn subset counts directly from set algebra.
        # This avoids relying on rendered label artists, which can be omitted for tiny regions.
        set_a, set_b, set_c = (tp_sets[set_keys[0]], tp_sets[set_keys[1]], tp_sets[set_keys[2]])
        overlap_counts = {
            '100': len(set_a - set_b - set_c),
            '010': len(set_b - set_a - set_c),
            '001': len(set_c - set_a - set_b),
            '110': len((set_a & set_b) - set_c),
            '101': len((set_a & set_c) - set_b),
            '011': len((set_b & set_c) - set_a),
            '111': len(set_a & set_b & set_c),
        }

        category_overlaps = {
            'A': overlap_counts['100'] + overlap_counts['110'] + overlap_counts['101'] + overlap_counts['111'],
            'B': overlap_counts['010'] + overlap_counts['110'] + overlap_counts['011'] + overlap_counts['111'],
            'C': overlap_counts['001'] + overlap_counts['101'] + overlap_counts['011'] + overlap_counts['111'],
        }
        print("Category overlaps (should match set sizes):", category_overlaps)
        if category_overlaps['A'] != len(set_a):
            print(f"Warning: Category A overlap ({category_overlaps['A']}) does not match set A size ({len(set_a)}).")
        else:
            print(f"✓ Category A overlap matches set A size: {category_overlaps['A']} records")
        if category_overlaps['B'] != len(set_b):
            print(f"Warning: Category B overlap ({category_overlaps['B']}) does not match set B size ({len(set_b)}).")
        else: 
            print(f"✓ Category B overlap matches set B size: {category_overlaps['B']} records")
        if category_overlaps['C'] != len(set_c):
            print(f"Warning: Category C overlap ({category_overlaps['C']}) does not match set C size ({len(set_c)}).")
        else:
            print(f"✓ Category C overlap matches set C size: {category_overlaps['C']} records")

        overlap_total = sum(overlap_counts.values())
        if overlap_total != total_unique_cnvs:
            print(
                f"Warning: overlap sum ({overlap_total}) does not match union total ({total_unique_cnvs})."
            )
        
        if len(set(truth_set_sizes.values())) > 1:
            print(f"Warning: TP+FN truth totals differ by method: {truth_set_sizes}")
        
        # Apply name mapping and add counts to labels
        display_names_with_counts = []
        for input_set_key in set_keys:
            count = len(tp_sets[input_set_key])
            pct_total = (count / total_unique_cnvs) * 100 if total_unique_cnvs > 0 else 0
            
            display_name = self.input_name_mapping.get(input_set_key, input_set_key)
            label = f"{display_name}\n(n={count}, {pct_total:.1f}%)"
            display_names_with_counts.append(label)
        
        # Create Venn diagram and optional side table for tiny/non-rendered regions
        if show_region_table:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3.2, 1.5])
            ax = fig.add_subplot(gs[0, 0])
            ax_table = fig.add_subplot(gs[0, 1])
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax_table = None
        
        display_names_tuple: Tuple[str, str, str] = (
            display_names_with_counts[0], 
            display_names_with_counts[1], 
            display_names_with_counts[2]
        )
        venn_subsets = (
            overlap_counts['100'],
            overlap_counts['010'],
            overlap_counts['110'],
            overlap_counts['001'],
            overlap_counts['101'],
            overlap_counts['011'],
            overlap_counts['111'],
        )

        venn_obj = venn3(
            subsets=venn_subsets,
            set_labels=display_names_tuple,
            ax=ax
        )
        venn_circles_obj = venn3_circles(subsets=venn_subsets, ax=ax)

        # Keep rendered labels synchronized with exact set-algebra counts.
        for region_id, count in overlap_counts.items():
            region_label = venn_obj.get_label_by_id(region_id)
            if region_label is not None:
                region_label.set_text(str(count))
        
        # Customize appearance
        for circle in venn_circles_obj:
            circle.set_linewidth(2)
            circle.set_linestyle('--')
        
        # Generate title
        svtype_str = f" ({svtype.value})" if svtype != SVType.ALL else ""
        title = f"Benchmark CNV Recall Overlap{svtype_str}\n({total_unique_cnvs} detected of {total_truth_cnvs} total truth records)"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Calculate statistics for text box
        recall_rate = (total_unique_cnvs / total_truth_cnvs * 100) if total_truth_cnvs > 0 else 0
        
        # Add summary statistics text box
        stats_text = (
            f"Total truth CNVs: {total_truth_cnvs} | "
            f"Detected by ≥1 method: {total_unique_cnvs} ({recall_rate:.1f}%) | "
            f"Not detected: {total_truth_cnvs - total_unique_cnvs}"
        )
        ax.text(
            0.5, -0.15, stats_text, 
            ha='center', 
            transform=ax.transAxes,
            fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

        # Matplotlib-Venn may suppress text for tiny non-zero regions.
        # This can make visual sums from displayed labels look too small.
        hidden_nonzero_regions = [
            region_id for region_id, count in overlap_counts.items()
            if count > 0 and venn_obj.get_label_by_id(region_id) is None
        ]
        if hidden_nonzero_regions:
            hidden_desc = ", ".join(
                f"{region_id}={overlap_counts[region_id]}" for region_id in hidden_nonzero_regions
            )
            print(
                "Warning: Non-zero Venn regions not displayed due to layout constraints: "
                f"{hidden_desc}"
            )

        if show_region_table and ax_table is not None:
            display_names = [
                self.input_name_mapping.get(set_keys[0], set_keys[0]),
                self.input_name_mapping.get(set_keys[1], set_keys[1]),
                self.input_name_mapping.get(set_keys[2], set_keys[2]),
            ]

            region_labels = {
                '100': f"{display_names[0]} only",
                '010': f"{display_names[1]} only",
                '001': f"{display_names[2]} only",
                '110': f"{display_names[0]} & {display_names[1]}",
                '101': f"{display_names[0]} & {display_names[2]}",
                '011': f"{display_names[1]} & {display_names[2]}",
                '111': f"{display_names[0]} & {display_names[1]} & {display_names[2]}",
            }

            # Keep lines compact and ordered by classic 3-set Venn order.
            region_order = ['100', '010', '001', '110', '101', '011', '111']
            table_lines = ["Region counts", ""]
            for region_id in region_order:
                count = overlap_counts[region_id]
                pct = (count / total_unique_cnvs) * 100 if total_unique_cnvs > 0 else 0.0
                table_lines.append(f"{region_labels[region_id]}")
                table_lines.append(f"  {region_id}: {count} ({pct:.1f}%)")

            if hidden_nonzero_regions:
                table_lines.append("")
                table_lines.append("Hidden in-plot labels:")
                table_lines.append(", ".join(hidden_nonzero_regions))

            ax_table.axis('off')
            ax_table.text(
                0.0,
                1.0,
                "\n".join(table_lines),
                transform=ax_table.transAxes,
                ha='left',
                va='top',
                fontsize=9,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9)
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
        print(f"Detected by at least one method: {total_unique_cnvs} ({recall_rate:.1f}% of truth)")
        print(f"Not detected by any method (FN): {total_truth_cnvs - total_unique_cnvs}")
        
        print(f"\nDetection by individual methods:")
        for input_set_key in set_keys:
            count = len(tp_sets[input_set_key])
            display_name = self.input_name_mapping.get(input_set_key, input_set_key)
            pct_truth = (count / total_truth_cnvs) * 100 if total_truth_cnvs > 0 else 0
            pct_detected = (count / total_unique_cnvs) * 100 if total_unique_cnvs > 0 else 0
            print(f"  {display_name}: {count} ({pct_truth:.1f}% of truth, {pct_detected:.1f}% of detected)")
        
        print(f"\nDetailed Overlap Counts:")
        combinations = ['100', '010', '001', '110', '101', '011', '111']
        
        for comb in combinations:
            count = overlap_counts[comb]
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
            pct_each_method_str = " | ".join(pct_each_method)
            
            print(f"  Combination {comb}: {count} ({detected:.1f}% of detected, {pct_truth:.1f}% of truth)")
            print(f"    - {pct_each_method_str}")

    def plot_size_distribution(
        self,
        input_sets_to_plot: Dict[str, List[str]],
        svtype: SVType = SVType.ALL,
        figsize: Tuple[int, int] = (12, 6),
        output_dir: Optional[str | Path] = None,
        include_benchmark: bool = True,
        stats_output_path: Optional[str | Path] = None,
    ):
        """
        Generate size distribution plots (bin density and KDE) for CNVs.
        
        Args:
            input_sets_to_plot: Dict mapping plot group name -> list of input set keys
            svtype: SV type to filter by (DEL, DUP, or ALL)
            figsize: Figure size tuple
            output_dir: Directory to save plots (if None, plots will be shown)
            include_benchmark: Whether to include benchmark truth set in plots
            stats_output_path: Path to save size summary stats table (csv/tsv)
        """

        input_sets = self.data.get('input_sets', {})
        input_sets_keys = set(input_sets.keys())

        # Create a string representation of the SV type for titles and filenames.
        svtype_str = f" ({svtype.value})" if svtype != SVType.ALL else ""

        # Prepare output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        plot_groups_count = len(input_sets_to_plot)
        generated_groups = 0
        group_membership_sets: Dict[str, set] = {}

        # 1) Collect all unique items across all input_sets_to_plot (preserve first-seen order).
        ordered_unique_input_sets: List[str] = []
        seen_input_sets = set()
        for input_set_list in input_sets_to_plot.values():
            for set_key in input_set_list:
                if set_key in input_sets_keys and set_key not in seen_input_sets:
                    seen_input_sets.add(set_key)
                    ordered_unique_input_sets.append(set_key)

        for plot_group_name, plot_group_input_sets in input_sets_to_plot.items():
            group_membership_sets[str(plot_group_name)] = {
                set_key for set_key in plot_group_input_sets if set_key in input_sets_keys
            }

        # 2) Calculate prediction data once and store in all_data.
        all_data = []
        for set_key in ordered_unique_input_sets:
            for classification in ['TP', 'FP']:
                df = input_sets.get(set_key, {}).get(classification)
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
                        'source': set_key,
                        'display_name': display_name,
                        'type': 'prediction'
                    })
                    all_data.append(temp_df)

        # 3) Retrieve benchmark data once from one prediction source and store in all_data.
        benchmark_set_key: Optional[str] = ordered_unique_input_sets[0] if ordered_unique_input_sets else None
        benchmark_added = False
        if include_benchmark and benchmark_set_key is not None:
            for classification in ['TP', 'FN']:
                df = input_sets.get(benchmark_set_key, {}).get(classification)
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
                    benchmark_added = True

        if not all_data:
            print("Warning: No data available for plotting.")
            return

        all_data_df = pd.concat(all_data, ignore_index=True)

        # Build a single summary stats table across all unique input sets.
        size_stats_df = (
            all_data_df
            .groupby('source', sort=True)['size']
            .apply(_compute_series_stats)
            .unstack()
            .reset_index()
            .rename(columns={'source': 'input_set'})
        )

        # Add boolean membership columns keyed by input_sets_to_plot dictionary keys.
        for plot_group_name, group_set in group_membership_sets.items():
            size_stats_df[plot_group_name] = (
                size_stats_df['input_set'].isin(group_set)
                | size_stats_df['input_set'].eq('Benchmark (Truth)')
            )

        # Keep benchmark at the bottom for readability.
        if not size_stats_df.empty:
            is_benchmark = size_stats_df['input_set'].eq('Benchmark (Truth)')
            size_stats_df = pd.concat(
                [size_stats_df[~is_benchmark], size_stats_df[is_benchmark]],
                ignore_index=True,
            )

        # Save or print one consolidated stats table.
        resolved_stats_output: Optional[Path] = None
        if stats_output_path:
            resolved_stats_output = Path(stats_output_path)
        elif output_dir:
            stats_suffix = svtype_str.replace(' ', '_')
            resolved_stats_output = output_dir / f"size_distribution_stats{stats_suffix}.tsv"

        if resolved_stats_output:
            resolved_stats_output.parent.mkdir(parents=True, exist_ok=True)
            file_ext = resolved_stats_output.suffix.lower()
            if file_ext in ['.tsv', '.txt']:
                size_stats_df.to_csv(resolved_stats_output, sep='\t', index=False)
            else:
                size_stats_df.to_csv(resolved_stats_output, index=False)
            print(f"✓ Size stats table saved to: {resolved_stats_output}")
        else:
            print("Size stats table (not saved; provide output_dir or stats_output_path):")
            print(size_stats_df.to_string(index=False))

        # 4) Iterate plot groups and pull only required items from all_data.
        for plot_group_name, plot_group_input_sets in input_sets_to_plot.items():
            plot_group_name_clean = str(plot_group_name).lower().replace(' ', '_').replace('/', '_')

            # Validate set_keys against data for this plot group.
            valid_group_input_sets = [set_key for set_key in plot_group_input_sets if set_key in input_sets_keys]
            if not valid_group_input_sets:
                print(f"Warning: No valid input sets found for plot group '{plot_group_name}'.")
                continue

            output_subdir: Optional[Path] = None
            if output_dir:
                output_subdir = output_dir / plot_group_name_clean
                output_subdir.mkdir(parents=True, exist_ok=True)

            plot_sources = set(valid_group_input_sets)
            if include_benchmark and benchmark_added:
                plot_sources.add('Benchmark (Truth)')

            plot_df = all_data_df[all_data_df['source'].isin(plot_sources)].copy()
            if plot_df.empty:
                print(f"Warning: No data available for plotting in plot group '{plot_group_name}'.")
                continue

            # Get min and max sizes for setting x-axis limits.
            min_size = plot_df['size'].min()
            max_size = plot_df['size'].max()

            title_suffix = " (with Benchmark)" if include_benchmark else ""
            title_prefix = f"{plot_group_name}: " if plot_group_name else ""

            # ==================== PLOT 1: BINNED DENSITY ====================
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
            ax.set_title(
                f"{title_prefix}CNV Size Distribution - Binned Density{svtype_str}{title_suffix}",
                fontsize=14,
                fontweight='bold'
            )

            # Customize legend
            legend = ax.get_legend()
            if legend:
                legend.set_title('Source')
                plt.setp(legend.get_texts(), fontsize=10)
                plt.setp(legend.get_title(), fontsize=10)

            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            if output_subdir:
                density_path = output_subdir / f"size_distribution_binned_density{svtype_str.replace(' ', '_')}.png"
                plt.savefig(density_path, dpi=300, bbox_inches='tight')
                print(f"✓ Binned density plot saved to: {density_path}")
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
            ax.set_title(
                f"{title_prefix}CNV Size Distribution - KDE{svtype_str}{title_suffix}",
                fontsize=14,
                fontweight='bold'
            )

            # Customize legend
            legend = ax.get_legend()
            if legend:
                legend.set_title('Source')
                plt.setp(legend.get_texts(), fontsize=10)
                plt.setp(legend.get_title(), fontsize=10)

            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            if output_subdir:
                kde_path = output_subdir / f"size_distribution_kde{svtype_str.replace(' ', '_')}.png"
                plt.savefig(kde_path, dpi=300, bbox_inches='tight')
                print(f"✓ KDE saved to: {kde_path}")
            else:
                plt.show()
            plt.close()

            generated_groups += 1

        if generated_groups == 0:
            print("Warning: No size distribution plots were generated.")
        else:
            print(f"✓ Size distribution plots completed for {generated_groups} plot group(s)!")

    def get_caller_source_distribution(
            self,
            input_sets_to_include: List[str], 
            output_file: Path
        ):
        """
        Analyze caller source distributions per sample and svtype, then generate box plots.
        
        Args:
            all_data: Dictionary of analysis data per input set
            output_dir: Directory to save plots
        """
        rows = []
        total_call_rows = []

        # Iterate only over input_sets (shared_FN is separate)
        input_sets = self.data.get('input_sets', {})
        for input_set_key, analysis_data in input_sets.items():
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

                        # Store one sample-level total call record per input_set/sample/svtype.
                        total_call_rows.append({
                            "input_set": input_set_key,
                            "sample": sample,
                            "svtype": svtype,
                            "total_calls": total_calls,
                        })
                        
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
                                "total_calls": total_calls,
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
                                "total_calls": total_calls,
                            })

        df = pd.DataFrame(rows)
        total_calls_df = pd.DataFrame(total_call_rows)
        
        if df.empty:
            print("No source distribution data found")
            return df
        
        print("\nCaller Source Distribution Summary:")
        print(df.head())
        
        # Custom sorting function: most combinations first (by count), then alphabetically
        def sort_key(entity):
            parts = entity.split("|")
            return (-len(parts), entity)
        
        # Preserve caller/input ordering from the function argument.
        # Only keep sets that are present in this dataframe.
        present_input_sets = set(df["input_set"].unique())
        input_sets = [s for s in input_sets_to_include if s in present_input_sets]
        num_input_sets = len(input_sets)
        
        # Create a single figure with 4 subplots (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(24, 14))
        
        # Define the layout: (row, col, svtype, metric, metric_label)
        plot_config = [
            (0, 0, "DEL", "raw_count", "Raw Caller"),
            (0, 1, "DEL", "combination_count", "Caller Combination"),
            (1, 0, "DUP", "raw_count", "Raw Caller"),
            (1, 1, "DUP", "combination_count", "Caller Combination"),
        ]
        
        # Get all unique entities across all subplots for consistent coloring
        all_entities_raw = set()
        all_entities_combo = set()
        for svtype in ["DEL", "DUP"]:
            for metric in ["raw_count", "combination_count"]:
                subset = df[(df["svtype"] == svtype) & (df["metric"] == metric)]
                if metric == "raw_count":
                    all_entities_raw.update(subset["caller_or_combination"].unique())
                else:
                    all_entities_combo.update(subset["caller_or_combination"].unique())
        
        # Sort entities
        entities_raw_sorted = sorted(all_entities_raw, key=sort_key)
        entities_combo_sorted = sorted(all_entities_combo, key=sort_key)
        
        # Create color mappings for entities (callers/combinations)
        colors_raw = matplotlib.colormaps['tab10'](np.linspace(0, 1, max(len(entities_raw_sorted), 1)))
        colors_combo = matplotlib.colormaps['tab20'](np.linspace(0, 1, max(len(entities_combo_sorted), 1)))
        
        color_map_raw = {entity: colors_raw[i] for i, entity in enumerate(entities_raw_sorted)}
        color_map_combo = {entity: colors_combo[i] for i, entity in enumerate(entities_combo_sorted)}
        
        # Create color mapping for input sets (different patterns/shades)
        input_set_colors = matplotlib.colormaps['Set2'](np.linspace(0, 1, num_input_sets))
        input_set_color_map = {input_set: input_set_colors[i] for i, input_set in enumerate(input_sets)}
        
        # Generate each subplot
        for row, col, svtype, metric, metric_label in plot_config:
            ax = axes[row, col]
            subset = df[(df["svtype"] == svtype) & (df["metric"] == metric)]
            
            if subset.empty:
                ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{svtype} {metric_label}", fontsize=12, fontweight='bold')
                continue
            
            # Get sorted entities for this subplot
            if metric == "raw_count":
                entities = [e for e in entities_raw_sorted if e in subset["caller_or_combination"].values]
                entity_color_map = color_map_raw
            else:
                entities = [e for e in entities_combo_sorted if e in subset["caller_or_combination"].values]
                entity_color_map = color_map_combo
            
            # Calculate positions for grouped box plots
            box_width = 0.8 / num_input_sets
            group_gap = 1.0
            
            all_positions = []
            all_entity_data = []
            all_colors = []
            
            # Prepare data grouped by entity, with input sets side-by-side
            for entity_idx, entity in enumerate(entities):
                base_position = entity_idx * group_gap + 1
                
                for input_set_idx, input_set in enumerate(input_sets):
                    entity_input_data = subset[
                        (subset["caller_or_combination"] == entity) & 
                        (subset["input_set"] == input_set)
                    ]["percentage"].values
                    
                    if len(entity_input_data) > 0:
                        position = base_position + (input_set_idx - (num_input_sets - 1) / 2) * box_width
                        all_positions.append(position)
                        all_entity_data.append(entity_input_data)
                        all_colors.append(input_set_color_map[input_set])
            
            if all_entity_data:
                # Create box plots
                bp = ax.boxplot(all_entity_data, positions=all_positions, widths=box_width * 0.8, 
                            patch_artist=True, showfliers=False)
                
                # Style boxes with input set colors
                for box, color in zip(bp['boxes'], all_colors):
                    box.set_facecolor(color)
                    box.set_alpha(0.7)
                    box.set_linewidth(1.5)
                
                # Overlay individual points with jitter
                for pos, vals, color in zip(all_positions, all_entity_data, all_colors):
                    jitter = np.random.normal(loc=0, scale=box_width * 0.15, size=len(vals))
                    ax.scatter(np.full(len(vals), pos) + jitter, vals, s=20, alpha=0.6, 
                            color='black', zorder=3)
            
            # Set x-axis labels at entity group centers
            entity_positions = [i * group_gap + 1 for i in range(len(entities))]
            ax.set_xticks(entity_positions)
            ax.set_xticklabels(entities, rotation=45, ha='right')
            ax.set_xlim(0.5, len(entities) * group_gap + 0.5)
            
            ax.set_title(f"{svtype} {metric_label}", fontsize=12, fontweight='bold')
            ax.set_ylabel("Percentage per Sample (%)")
            ax.set_xlabel("Caller" if metric == "raw_count" else "Caller Combination")
            ax.grid(axis='y', alpha=0.3, linestyle='--')

        
        # Add a single, color-matched legend for average calls/sample per input set.
        if not total_calls_df.empty:
            avg_calls_by_set_sv = (
                total_calls_df
                .groupby(["input_set", "svtype"]) ["total_calls"]
                .mean()
                .to_dict()
            )

            avg_legend_elements = []
            for input_set in input_sets:
                del_avg = avg_calls_by_set_sv.get((input_set, "DEL"))
                dup_avg = avg_calls_by_set_sv.get((input_set, "DUP"))

                del_txt = f"DEL={del_avg:.1f}" if del_avg is not None else "DEL=NA"
                dup_txt = f"DUP={dup_avg:.1f}" if dup_avg is not None else "DUP=NA"
                label = f"{self.input_name_mapping.get(input_set, input_set)}\n{del_txt}\n{dup_txt}"

                avg_legend_elements.append(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        fc=input_set_color_map[input_set],
                        alpha=0.7,
                        label=label,
                    )
                )

            fig.legend(
                handles=avg_legend_elements,
                loc='upper center',
                ncol=num_input_sets,
                bbox_to_anchor=(0.5, 0.96),
                fontsize=9,
                frameon=True,
                title="Avg calls/sample by input set",
                title_fontsize=9,
            )
        
        plt.suptitle("Caller Source Distribution by Input Set", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=(0, 0.02, 1, 0.86))
        
        os.makedirs(output_file.parent, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_file} with caller source distribution box plots")
        
        return df

    def get_count_statistics(
        self, 
        svtype: SVType = SVType.ALL,
        output_file: Optional[str | Path] = None,
    ) -> Dict[str, dict]:

        # Summarize per-sample call counts for prediction (TP+FP) and benchmark (TP+FN).
        input_sets = self.data.get('input_sets', {})
        table_statistics: Dict[str, dict] = {}

        def _summarize_sample_counts(sample_counts: pd.Series) -> Dict[str, float | int]:
            stats = _compute_series_stats(sample_counts)
            return {
                'n_samples': int(stats['count']),
                'total_calls': int(sample_counts.sum()),
                'mean_calls_per_sample': float(stats['mean']),
                'median_calls': float(stats['median']),
                'min_calls': float(stats['min']),
                'max_calls': float(stats['max']),
                'q1_calls': float(stats['q1']),
                'q3_calls': float(stats['q3']),
                'iqr_calls': float(stats['iqr']),
            }

        for input_set_key, input_set_data in input_sets.items():
            tp_df = _filter_svtype(input_set_data.get('TP', pd.DataFrame()), svtype)
            fp_df = _filter_svtype(input_set_data.get('FP', pd.DataFrame()), svtype)

            # Prediction calls per sample: TP + FP
            pred_frames = [df for df in [tp_df, fp_df] if not df.empty and 'sample' in df.columns]
            if pred_frames:
                pred_sample_counts = pd.concat(pred_frames, ignore_index=True).groupby('sample').size()
            else:
                pred_sample_counts = pd.Series(dtype='int64')

            table_statistics[input_set_key] = {
                'display_name': self.input_name_mapping.get(input_set_key, input_set_key),
                'svtype': svtype.value if svtype != SVType.ALL else SVType.ALL.value,
                'input_set': input_set_key,
                'call_definition': 'TP+FP',
                **_summarize_sample_counts(pred_sample_counts),
            }

        # Build one benchmark row from the first available input set and append it last.
        first_input_set = next(iter(input_sets.items()), None)
        if first_input_set is not None:
            first_key, first_data = first_input_set
            first_tp_df = _filter_svtype(first_data.get('TP', pd.DataFrame()), svtype)
            first_fn_df = _filter_svtype(first_data.get('FN', pd.DataFrame()), svtype)

            benchmark_frames = [
                df for df in [first_tp_df, first_fn_df]
                if not df.empty and 'sample' in df.columns
            ]
            if benchmark_frames:
                benchmark_sample_counts = pd.concat(benchmark_frames, ignore_index=True).groupby('sample').size()
            else:
                benchmark_sample_counts = pd.Series(dtype='int64')

            table_statistics['benchmark'] = {
                'display_name': 'Benchmark (Truth)',
                'svtype': svtype.value if svtype != SVType.ALL else SVType.ALL.value,
                'input_set': 'benchmark',
                'call_definition': 'TP+FN',
                **_summarize_sample_counts(benchmark_sample_counts),
            }

        if output_file is not None:
            resolved_output = Path(output_file)
            resolved_output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(list(table_statistics.values())).to_csv(resolved_output, sep='\t', index=False)
            print(f"Count statistics saved to: {resolved_output}")

        return table_statistics