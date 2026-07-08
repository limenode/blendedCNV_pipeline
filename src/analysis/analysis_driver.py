from pathlib import Path
import json
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from utils import parse_args, PipelineConfig, f0_5_score, f2_score, precision, recall, f1_score, SVType
from analysis.cnv_plotter import CNVPlotter
from analysis.analysis_functions import load_data_for_all_input_sets, get_samples_from_data, analyze_logs, get_counts_from_config


def _load_plots_config(config: PipelineConfig) -> dict:
    plots_config_path = config.analysis_plots_config
    if not plots_config_path:
        print("No analysis_plots_config defined in configuration. Skipping plot generation.")
        return {}
    path = Path(plots_config_path)
    if not path.exists():
        print(f"Warning: analysis_plots_config file not found at '{plots_config_path}'. Skipping plot generation.")
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main(config: PipelineConfig):
    if not config.experimental:
        print("No input sets defined in configuration. Exiting analysis pipeline.")
        return

    if not config.benchmark:
        print("No benchmark map defined in configuration. Skipping benchmark analysis.")
        return

    # === Step 1: Prepare Input Set Paths ===

    # Create a mapping of input set keys to their corresponding paths
    input_sets_paths = {}

    # Setup input name mapping for user-friendly display
    input_name_mapping = {}

    # Get all input set keys
    input_sets_raw = list(config.experimental.keys())
    print(f"Available input sets: {input_sets_raw}")

    # Append each caller, "Intersection", and "Union" to input set keys for binary classification results
    output_dir = config.output_dir
    layout = config.layout
    for key in input_sets_raw:
        key_path = key.replace(" ", "_")
        input_set_subdir = layout.classification_root(key)

        for caller in config.experimental[key].keys():
            caller_key = f"{key_path}_{caller}"
            input_sets_paths[caller_key] = input_set_subdir / caller
            input_name_mapping[caller_key] = f"{key} {caller}"

        for consensus_type in ['consensus_1of3', 'consensus_2of3', 'consensus_3of3']:
            for merge_type in ['intersections', 'unions']:
                consensus_key = f"{key_path}_{consensus_type}_{merge_type}"
                input_sets_paths[consensus_key] = input_set_subdir / f"{consensus_type}_{merge_type}"
                input_name_mapping[consensus_key] = f"{key} {consensus_type.replace('_', ' ').title()} {merge_type.title()}"

        # Add to input name mapping
        input_name_mapping[key_path + "_intersections"] = f"{key} Intersections"
        input_name_mapping[key_path + "_unions"] = f"{key} Unions"

    # Append control sets
    control_sets_raw = list(config.control.keys())
    for key in control_sets_raw:
        key_path = key.replace(" ", "_")
        input_sets_paths[key_path] = layout.control_classification_dir(key)
        input_name_mapping[key_path] = key


    # === Load plots config ===
    plots_config = _load_plots_config(config)
    if not plots_config:
        return


    # === Log Analysis Step 1: Load logs ===
    log_dir = layout.logs
    samples_of_interest = plots_config.get('samples_of_interest', [])
    analyze_logs(log_dir, output_dir=output_dir, samples=samples_of_interest)



    # === Step 2: Load Data for All Input Sets ===
    all_data = load_data_for_all_input_sets(input_sets_paths)
    samples = get_samples_from_data(all_data, classification_key='TP')

    counts_tuple = get_counts_from_config(config, bounds=(500, 1_000_000), samples=list(samples))
    counts_tuple_all = get_counts_from_config(config, samples=list(samples))

    # Dump counts to JSON for record-keeping
    counts_output_path = layout.logs / "analysis_counts_summary.json"
    with open(counts_output_path, 'w') as f:
        json.dump(counts_tuple, f, indent=4)
    print(f"\nSaved counts summary to {counts_output_path}")

    counts_output_path_all = layout.logs / "analysis_counts_summary_all.json"
    with open(counts_output_path_all, 'w') as f:
        json.dump(counts_tuple_all, f, indent=4)
    print(f"\nSaved unfiltered counts summary to {counts_output_path_all}")

    plotter = CNVPlotter(all_data, config, input_name_mapping)
    metrics = [(precision, "Precision"), (recall, "Recall"), (f1_score, "F1 Score"), (f0_5_score, "F 1/2 Score"), (f2_score, "F2 Score")]

    # Build venn_diagram_specs from plots config, inserting output paths
    venn_diagrams_raw = plots_config.get('venn_diagrams', {})
    venn_diagram_specs = {
        name: {
            "set_keys": spec['sets'],
            "output_path": layout.venn_figures / spec['output_filename'],
            "title": spec['title'],
        }
        for name, spec in venn_diagrams_raw.items()
    }

    statistical_distribution_input_sets = plots_config.get('statistical_distributions', {})
    size_distribution_input_sets = plots_config.get('size_distributions', {})
    statistical_distribution_split_by_svtype_sets = plots_config.get('statistical_distributions_split_by_svtype_sets', {})
    

    caller_source_sets = plots_config.get('caller_source_distribution', {}).get('sets', [])
    count_venn_input_names = plots_config.get('count_venn_diagrams', [])

    # === Step 3-5: Generate All Plots in Parallel ===
    # Define all plotting tasks as partial functions for parallel execution
    plotting_tasks = [
        # Task 1: Statistical distributions for all SV types only
        partial(
            plotter.plot_statistical_distributions,
            plot_config=statistical_distribution_input_sets,
            metrics=metrics,
            svtypes=[SVType.ALL],
            bounds=(500, 1_000_000),
            output_dir=layout.stat_dist_all_figures,
            cumulative_stats_output_path=layout.log("statistical_distributions_cumulative_stats.tsv"),
        ),
        # Task 1b: Statistical distributions split by SV type
        partial(
            plotter.plot_statistical_distributions,
            plot_config=statistical_distribution_split_by_svtype_sets,
            metrics=metrics,
            svtypes=[SVType.DEL, SVType.DUP],
            bounds=(500, 1_000_000),
            output_dir=layout.stat_dist_split_figures,
            cumulative_stats_output_path=layout.log("statistical_distributions_split_by_svtype_cumulative_stats.tsv"),
        ),  
        # Task 2: Size distribution plots
        partial(
            plotter.plot_size_distribution,
            plot_config=size_distribution_input_sets,
            output_dir=layout.size_figures,
            include_benchmark=True,
            stats_output_path=layout.log("size_distribution_stats.tsv"),
        ),
        # Task 2b: Size distribtion plots mod 3000
        partial(
            plotter.plot_size_distribution,
            plot_config=size_distribution_input_sets,
            output_dir=layout.size_figures,
            include_benchmark=True,
            stats_output_path=layout.log("size_distribution_stats.tsv"),
            modulus=3000,
        ),
        # Task 3: Caller source distribution
        partial(
            plotter.get_caller_source_distribution,
            input_sets_to_include=caller_source_sets,
            output_file=layout.caller_source_figures / "caller_source_distribution.png",
        ),
        # Task 4: Get counts
        partial(
            plotter.get_count_statistics,
            output_file=layout.log("count_statistics.tsv"),
        ),
    ]

    # Append Venn diagram plotting tasks
    for spec in venn_diagram_specs.values():
        plotting_tasks.append(
            partial(
                plotter.plot_recall_venn_diagram,
                set_keys=spec['set_keys'],
                output_path=spec['output_path'],
            )
        )

    for input_name in count_venn_input_names:
        plotting_tasks.append(
            partial(
                plotter.plot_count_venn_diagram,
                config=config,
                input_set_key=input_name,
                output_path=layout.venn_figures / f"count_venn_diagram_{input_name.replace(' ', '_')}.png",
            )
        )

    # Execute plotting tasks in parallel
    print(f"\nExecuting {len(plotting_tasks)} plotting tasks in parallel...")
    with ProcessPoolExecutor(max_workers=len(plotting_tasks)) as executor:
        futures = [executor.submit(task) for task in plotting_tasks]

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                future.result()
                print(f"Completed plotting task {completed}/{len(plotting_tasks)}")
            except Exception as e:
                print(f"Error in plotting task: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    # Allow running standalone for testing
    config = parse_args()

    main(config)
