"""Analysis phase: metrics, counts, and plots over the classified call sets."""

from consensuscnv.analysis.analysis_driver import main as run_analysis
from consensuscnv.analysis.analysis_functions import (
    analyze_logs,
    get_bed_counts,
    get_counts_from_config,
    get_samples_from_data,
    load_data_for_all_queries,
    plot_excluded_regions_violin_plots,
    plot_liftover_results,
)
from consensuscnv.analysis.cnv_plotter import CNVPlotter, identify_undiscoverable_cnvs
from consensuscnv.analysis.load_analysis_data import (
    build_analysis_data_structure,
    discover_classification_files,
    filter_by_size,
    load_fn_file,
    load_fp_file,
    load_sample_data,
    load_tp_file,
)

__all__ = [
    "CNVPlotter",
    "analyze_logs",
    "build_analysis_data_structure",
    "discover_classification_files",
    "filter_by_size",
    "get_bed_counts",
    "get_counts_from_config",
    "get_samples_from_data",
    "identify_undiscoverable_cnvs",
    "load_data_for_all_queries",
    "load_fn_file",
    "load_fp_file",
    "load_sample_data",
    "load_tp_file",
    "plot_excluded_regions_violin_plots",
    "plot_liftover_results",
    "run_analysis",
]
