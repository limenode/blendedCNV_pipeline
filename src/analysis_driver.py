from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from utils import parse_args, f0_5_score, f2_score, precision, recall, f1_score, SVType
from cnv_plotter import CNVPlotter
from analysis_functions import load_data_for_all_input_sets, get_samples_from_data, analyze_logs, get_counts_from_config

def main(config: dict):
    """
    Main analysis pipeline.
    
    Args:
        config: Configuration dictionary loaded from YAML
    """

    if 'input' not in config:
        print("No input sets defined in configuration. Exiting analysis pipeline.")
        return
    
    if 'benchmark_map' not in config:
        print("No benchmark map defined in configuration. Skipping benchmark analysis.")
        return

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
    control_sets_raw = list(config.get('control', {}).keys())
    for key in control_sets_raw:
        key_path = key.replace(" ", "_")
        input_sets_paths[key_path] = output_dir / key_path / "binary_classification"
        input_name_mapping[key_path] = key    

    # === Log Analysis Step 1: Load logs ===
    log_dir = Path(config['output_dir']) / "logs"
    samples_of_interest = ['HG01890', 'NA19347', 'HG00513', 'HG01596', 'NA19238', 'NA19331', 'HG00096', 'HG00171', 'NA18989', 'HG00268', 'NA20847', 'HG00731', 'NA19129']
    analyze_logs(log_dir, output_dir=output_dir, samples=samples_of_interest)

    # === Step 2: Load Data for All Input Sets ===
    all_data = load_data_for_all_input_sets(input_sets_paths)
    samples = get_samples_from_data(all_data, classification_key='TP')

    counts_tuple = get_counts_from_config(config, bounds=(500, 1_000_000), samples=list(samples))
    counts_tuple_all = get_counts_from_config(config, samples=list(samples))

    # Dump counts to JSON for record-keeping
    counts_output_path = output_dir / "analysis_counts_summary.json"
    with open(counts_output_path, 'w') as f:
        json.dump(counts_tuple, f, indent=4)
    print(f"\nSaved counts summary to {counts_output_path}")

    counts_output_path_all = output_dir / "analysis_counts_summary_all.json"
    with open(counts_output_path_all, 'w') as f:
        json.dump(counts_tuple_all, f, indent=4)
    print(f"\nSaved unfiltered counts summary to {counts_output_path_all}")
    
    plotter = CNVPlotter(all_data, config, input_name_mapping)
    metrics = [(precision, "Precision"), (recall, "Recall"), (f1_score, "F1 Score"), (f0_5_score, "F 1/2 Score"), (f2_score, "F2 Score")]

    # Venn Diagram specifications:
    venn_diagram_specs = {
        "6x_intersections": {
            "set_keys": ['6x_Coverage_intersections', '30x_Coverage_intersections', 'SNP_Array'],
            "output_path": output_dir / "figures" / "venn_diagrams" / "venn_diagram_6x_intersections.png",
            "title": "Venn Diagram of Intersections with 6x Coverage"
        },

        "4x_intersections": {
            "set_keys": ['4x_Coverage_intersections', '30x_Coverage_intersections', 'SNP_Array'],
            "output_path": output_dir / "figures" / "venn_diagrams" / "venn_diagram_4x_intersections.png",
            "title": "Venn Diagram of Intersections with 4x Coverage"
        },

        "2x_intersections": {
            "set_keys": ['2x_Coverage_intersections', '30x_Coverage_intersections', 'SNP_Array'],
            "output_path": output_dir / "figures" / "venn_diagrams" / "venn_diagram_2x_intersections.png",
            "title": "Venn Diagram of Intersections with 2x Coverage"
        },
        "6x_unions": {
            "set_keys": ['6x_Coverage_unions', '30x_Coverage_unions', 'SNP_Array'],
            "output_path": output_dir / "figures" / "venn_diagrams" / "venn_diagram_6x_unions.png",
            "title": "Venn Diagram of Unions with 6x Coverage"
        },
        "4x_unions": {
            "set_keys": ['4x_Coverage_unions', '30x_Coverage_unions', 'SNP_Array'],
            "output_path": output_dir / "figures" / "venn_diagrams" / "venn_diagram_4x_unions.png",
            "title": "Venn Diagram of Unions with 4x Coverage"
        },
        "2x_unions": {
            "set_keys": ['2x_Coverage_unions', '30x_Coverage_unions', 'SNP_Array'],
            "output_path": output_dir / "figures" / "venn_diagrams" / "venn_diagram_2x_unions.png",
            "title": "Venn Diagram of Unions with 2x Coverage"
        }
    }

    # Size Distribution Set Keys
    size_distribution_set_keys = [key for key in all_data['input_sets'].keys() if "intersections" in key]
    size_distribution_set_keys.append("SNP_Array")


    # === Step 3-5: Generate All Plots in Parallel ===
    # Define all plotting tasks as partial functions for parallel execution
    plotting_tasks = [
        # Task 1: Statistical distributions
        partial(
            plotter.plot_statistical_distributions,
            metrics=metrics,
            bounds=(500, 1_000_000),
            output_path=output_dir / "figures" / "statistical_distributions" / "distribution.png",
        ),
        # # Task 1.5: Statistical distributions for all SV types only
        partial(
            plotter.plot_statistical_distributions,
            metrics=metrics,
            svtypes=[SVType.ALL],
            bounds=(500, 1_000_000),
            output_path=output_dir / "figures" / "statistical_distributions_all_only" / "distribution.png",
        ),
        # Task 2: Size distribution plots
        partial(
            plotter.plot_size_distribution,
            set_keys=size_distribution_set_keys,
            output_dir=output_dir / "figures" / "size_distributions",
            include_benchmark=True,
        ),
        # Task 3: Caller source distribution
        partial(
            plotter.get_caller_source_distribution,
            input_sets_to_include=[key for key in all_data['input_sets'].keys() if "intersections" in key],
            output_file=output_dir / "figures" / "caller_source_distribution" / "caller_source_distribution.png",
        ),
    ]

    # Append Venn diagram plotting tasks
    for spec in venn_diagram_specs.values():
        plotting_tasks.append(
            partial(
                plotter.plot_venn_diagram,
                set_keys=spec['set_keys'],
                output_path=spec['output_path'],
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

