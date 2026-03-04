from pathlib import Path
import yaml
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from utils import parse_args, f0_5_score, f2_score, precision, recall, f1_score
from cnv_plotter import CNVPlotter
from analysis_functions import load_data_for_all_input_sets, get_samples_from_data, analyze_logs, get_counts_from_config

def main(config: dict, debug: bool = False):
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

    time_0 = time.time()

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

    time_1 = time.time()

    # === Log Analysis Step 1: Load logs ===
    log_dir = Path(config['output_dir']) / "logs"
    samples_of_interest = ['HG01890', 'NA19347', 'HG00513', 'HG01596', 'NA19238', 'NA19331', 'HG00096', 'HG00171', 'NA18989', 'HG00268', 'NA20847', 'HG00731', 'NA19129']
    analyze_logs(log_dir, output_dir=output_dir, samples=samples_of_interest)

    time_2 = time.time()

    # === Step 2: Load Data for All Input Sets ===
    all_data = load_data_for_all_input_sets(input_sets_paths)
    all_data_no_filter = load_data_for_all_input_sets(input_sets_paths, bounds=(0, 1_000_000_000))
    samples = get_samples_from_data(all_data, classification_key='TP')

    res = get_counts_from_config(config, bounds=(500, 1_000_000), samples=list(samples))
    res_all = get_counts_from_config(config, samples=list(samples))

    # Dump counts to JSON for record-keeping
    counts_output_path = output_dir / "analysis_counts_summary.json"
    with open(counts_output_path, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"\nSaved counts summary to {counts_output_path}")

    counts_output_path_all = output_dir / "analysis_counts_summary_all.json"
    with open(counts_output_path_all, 'w') as f:
        json.dump(res_all, f, indent=4)
    print(f"\nSaved unfiltered counts summary to {counts_output_path_all}")

    # time_3 = time.time()

    # # === Print summary of loaded data ===
    # counts_dict = get_counts_from_data(all_data_no_filter)
    # counts_dict_no_filter = get_counts_from_data(all_data_no_filter)
    
    # # Dump counts to JSON for record-keeping
    # counts_output_path = output_dir / "analysis_counts_summary.json"
    # with open(counts_output_path, 'w') as f:
    #     json.dump({
    #         "filtered_counts": counts_dict,
    #         "unfiltered_counts": counts_dict_no_filter
    #     }, f, indent=4)
    # print(f"\nSaved counts summary to {counts_output_path}")
    
    # plotter = CNVPlotter(all_data, config, input_name_mapping)
    # metrics = [(precision, "Precision"), (recall, "Recall"), (f1_score, "F1 Score"), (f0_5_score, "F 1/2 Score"), (f2_score, "F2 Score")]

    # # === Step 3-5: Generate All Plots in Parallel ===
    # # Define all plotting tasks as partial functions for parallel execution
    # plotting_tasks = [
    #     # Task 1: Statistical distributions
    #     partial(
    #         plotter.plot_statistical_distributions,
    #         metrics=metrics,
    #         bounds=(500, 1_000_000),
    #         output_path=output_dir / "figures" / "statistical_distributions" / "distribution.png",
    #     ),
    #     # Task 2: Venn diagram for intersections
    #     partial(
    #         plotter.plot_venn_diagram,
    #         set_keys=['Low_Coverage_intersections', 'High_Coverage_intersections', 'SNP_Array'],
    #         output_path=output_dir / "figures" / "venn_diagrams" / "venn_diagram_intersections.png",
    #     ),
    #     # Task 3: Venn diagram for unions
    #     partial(
    #         plotter.plot_venn_diagram,
    #         set_keys=['Low_Coverage_unions', 'High_Coverage_unions', 'SNP_Array'],
    #         output_path=output_dir / "figures" / "venn_diagrams" / "venn_diagram_unions.png",
    #     ),
    #     # Task 4: Size distribution plots
    #     partial(
    #         plotter.plot_size_distribution,
    #         set_keys=list(all_data['input_sets'].keys()),
    #         output_dir=output_dir / "figures" / "size_distributions",
    #     ),
    #     # Task 5: Caller source distribution
    #     partial(
    #         plotter.get_caller_source_distribution,
    #         input_sets_to_include=[key for key in all_data['input_sets'].keys() if "intersections" in key],
    #         output_file=output_dir / "figures" / "caller_source_distribution" / "caller_source_distribution.png",
    #     ),
    # ]

    # # Execute plotting tasks in parallel
    # print(f"\nExecuting {len(plotting_tasks)} plotting tasks in parallel...")
    # with ProcessPoolExecutor(max_workers=len(plotting_tasks)) as executor:
    #     futures = [executor.submit(task) for task in plotting_tasks]
        
    #     # Collect results as they complete
    #     completed = 0
    #     for future in as_completed(futures):
    #         completed += 1
    #         try:
    #             future.result()
    #             print(f"Completed plotting task {completed}/{len(plotting_tasks)}")
    #         except Exception as e:
    #             print(f"Error in plotting task: {e}")
    #             import traceback
    #             traceback.print_exc()

    # time_4 = time.time()

    # if debug:
    #     print("\n=== DEBUG TIMING INFO ===")
    #     print(f"Input set path preparation time: {time_1 - time_0:.2f} seconds")
    #     print(f"Log analysis (loading) time: {time_2 - time_1:.2f} seconds")
    #     print(f"Data loading time: {time_3 - time_2:.2f} seconds")
    #     print(f"Parallel plotting time: {time_4 - time_3:.2f} seconds")
    #     print(f"Total analysis pipeline time: {time_4 - time_0:.2f} seconds")


if __name__ == "__main__":
    # Allow running standalone for testing
    config = parse_args()
    
    main(config, debug=True)

