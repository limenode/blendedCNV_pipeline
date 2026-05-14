from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from utils import parse_args, f0_5_score, f2_score, precision, recall, f1_score, SVType
from analysis.cnv_plotter import CNVPlotter
from analysis.analysis_functions import load_data_for_all_input_sets, get_samples_from_data, analyze_logs, get_counts_from_config

def main(config: dict):
    if 'input' not in config:
        print("No input sets defined in configuration. Exiting analysis pipeline.")
        return
    
    if 'benchmark_map' not in config:
        print("No benchmark map defined in configuration. Skipping benchmark analysis.")
        return

    # === Step 1: Prepare Input Set Paths ===

    # Create a mapping of input set keys to their corresponding paths
    input_sets_paths = {}

    # Setup input name mapping for user-friendly display
    input_name_mapping = {}

    # Get all input set keys
    input_sets_raw = list(config['input'].keys())
    print(f"Available input sets: {input_sets_raw}")

    # Append each caller, "Intersection", and "Union" to input set keys for binary classification results
    output_dir = Path(config['output_dir'])
    for key in input_sets_raw:
        key_path = key.replace(" ", "_")
        input_set_subdir = output_dir / key_path / "binary_classification"

        for caller in config['input'][key].keys():
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
            "set_keys": ['6x_Coverage_consensus_2of3_intersections', '30x_Coverage_consensus_2of3_intersections', 'SNP_Array'],
            "output_path": output_dir / "figures" / "venn_diagrams" / "venn_diagram_6x_intersections.png",
            "title": "Venn Diagram of Intersections with 6x Coverage"
        },

        "4x_intersections": {
            "set_keys": ['4x_Coverage_consensus_2of3_intersections', '30x_Coverage_consensus_2of3_intersections', 'SNP_Array'],
            "output_path": output_dir / "figures" / "venn_diagrams" / "venn_diagram_4x_intersections.png",
            "title": "Venn Diagram of Intersections with 4x Coverage"
        },

        "2x_intersections": {
            "set_keys": ['2x_Coverage_consensus_2of3_intersections', '30x_Coverage_consensus_2of3_intersections', 'SNP_Array'],
            "output_path": output_dir / "figures" / "venn_diagrams" / "venn_diagram_2x_intersections.png",
            "title": "Venn Diagram of Intersections with 2x Coverage"
        },
        "6x_unions": {
            "set_keys": ['6x_Coverage_consensus_2of3_unions', '30x_Coverage_consensus_2of3_unions', 'SNP_Array'],
            "output_path": output_dir / "figures" / "venn_diagrams" / "venn_diagram_6x_unions.png",
            "title": "Venn Diagram of Unions with 6x Coverage"
        },
        "4x_unions": {
            "set_keys": ['4x_Coverage_consensus_2of3_unions', '30x_Coverage_consensus_2of3_unions', 'SNP_Array'],
            "output_path": output_dir / "figures" / "venn_diagrams" / "venn_diagram_4x_unions.png",
            "title": "Venn Diagram of Unions with 4x Coverage"
        },
        "2x_unions": {
            "set_keys": ['2x_Coverage_consensus_2of3_unions', '30x_Coverage_consensus_2of3_unions', 'SNP_Array'],
            "output_path": output_dir / "figures" / "venn_diagrams" / "venn_diagram_2x_unions.png",
            "title": "Venn Diagram of Unions with 2x Coverage"
        },
    }

    # Input sets to plot
    statistical_distribution_input_sets = {
        "30x": {
            "sets": ['30x_Coverage_consensus_1of3_intersections', '30x_Coverage_consensus_1of3_unions', 
                '30x_Coverage_consensus_2of3_intersections', '30x_Coverage_consensus_2of3_unions', 
                '30x_Coverage_consensus_3of3_intersections', '30x_Coverage_consensus_3of3_unions',
                '30x_Coverage_cnvpytor', '30x_Coverage_gatk', '30x_Coverage_delly', 'SNP_Array'],
            "title": "Statistical Distributions for 30x Coverage Call Sets"
        },
        "6x": {
            "sets": ['6x_Coverage_consensus_1of3_intersections', '6x_Coverage_consensus_1of3_unions',
               '6x_Coverage_consensus_2of3_intersections', '6x_Coverage_consensus_2of3_unions',
               '6x_Coverage_consensus_3of3_intersections', '6x_Coverage_consensus_3of3_unions',
               '6x_Coverage_cnvpytor', '6x_Coverage_gatk', '6x_Coverage_delly', 'SNP_Array'],
            "title": "Statistical Distributions for 6x Coverage Call Sets"
        },
        "4x": {
            "sets": ['4x_Coverage_consensus_1of3_intersections', '4x_Coverage_consensus_1of3_unions',
                        '4x_Coverage_consensus_2of3_intersections', '4x_Coverage_consensus_2of3_unions',
                        '4x_Coverage_consensus_3of3_intersections', '4x_Coverage_consensus_3of3_unions',
                        '4x_Coverage_cnvpytor', '4x_Coverage_gatk', '4x_Coverage_delly', 'SNP_Array'],
            "title": "Statistical Distributions for 4x Coverage Call Sets"
        },
        "2x": {
            "sets": ['2x_Coverage_consensus_1of3_intersections', '2x_Coverage_consensus_1of3_unions',
                       '2x_Coverage_consensus_2of3_intersections', '2x_Coverage_consensus_2of3_unions',
                       '2x_Coverage_consensus_3of3_intersections', '2x_Coverage_consensus_3of3_unions',
                       '2x_Coverage_cnvpytor', '2x_Coverage_gatk', '2x_Coverage_delly', 'SNP_Array'],
            "title": "Statistical Distributions for 2x Coverage Call Sets"
        },
        "30x_no_unions": {
            "sets": ['30x_Coverage_consensus_1of3_intersections', '30x_Coverage_consensus_2of3_intersections',
                      '30x_Coverage_consensus_3of3_intersections', '30x_Coverage_cnvpytor', 
                      '30x_Coverage_gatk', '30x_Coverage_delly', 'SNP_Array'],
            "title": "Statistical Distributions for 30x Coverage Call Sets"
        },
        "6x_no_unions": {
            "sets": ['6x_Coverage_consensus_1of3_intersections', '6x_Coverage_consensus_2of3_intersections',
                      '6x_Coverage_consensus_3of3_intersections', '6x_Coverage_cnvpytor', 
                      '6x_Coverage_gatk', '6x_Coverage_delly', 'SNP_Array'],
            "title": "Statistical Distributions for 6x Coverage Call Sets"
        },
        "4x_no_unions": {
            "sets": ['4x_Coverage_consensus_1of3_intersections', '4x_Coverage_consensus_2of3_intersections',
                      '4x_Coverage_consensus_3of3_intersections', '4x_Coverage_cnvpytor', 
                      '4x_Coverage_gatk', '4x_Coverage_delly', 'SNP_Array'],
            "title": "Statistical Distributions for 4x Coverage Call Sets"
        },
        "2x_no_unions": {
            "sets": ['2x_Coverage_consensus_1of3_intersections', '2x_Coverage_consensus_2of3_intersections',
                      '2x_Coverage_consensus_3of3_intersections', '2x_Coverage_cnvpytor', 
                      '2x_Coverage_gatk', '2x_Coverage_delly', 'SNP_Array'],
            "title": "Statistical Distributions for 2x Coverage Call Sets"
        },
        "Intersections": {
            "sets": ['30x_Coverage_consensus_2of3_intersections', '6x_Coverage_consensus_2of3_intersections',
                      '4x_Coverage_consensus_2of3_intersections', '2x_Coverage_consensus_2of3_intersections', 'SNP_Array'],
            "title": "Statistical Distributions for Intersection Call Sets"
        },
        "Unions": {
            "sets": ['30x_Coverage_consensus_2of3_unions', '6x_Coverage_consensus_2of3_unions',
                      '4x_Coverage_consensus_2of3_unions', '2x_Coverage_consensus_2of3_unions', 'SNP_Array'],
            "title": "Statistical Distributions for Union Call Sets"
        },
        "CNVpytor": {
            "sets": ['30x_Coverage_cnvpytor', '6x_Coverage_cnvpytor', 
                     '4x_Coverage_cnvpytor', '2x_Coverage_cnvpytor', 'SNP_Array'],
            "title": "Statistical Distributions for CNVpytor Call Sets"
        },
        "GATK": {
            "sets": ['30x_Coverage_gatk', '6x_Coverage_gatk', 
                     '4x_Coverage_gatk', '2x_Coverage_gatk', 'SNP_Array'],
            "title": "Statistical Distributions for GATK Call Sets"
        },
        "DELLY": {
            "sets": ['30x_Coverage_delly', '6x_Coverage_delly', 
                     '4x_Coverage_delly', '2x_Coverage_delly', 'SNP_Array'],
            "title": "Statistical Distributions for DELLY Call Sets"
        },
    }

    size_distribution_input_sets = {
                "30x": {
            "sets": ['30x_Coverage_consensus_1of3_intersections', '30x_Coverage_consensus_1of3_unions', 
                '30x_Coverage_consensus_2of3_intersections', '30x_Coverage_consensus_2of3_unions', 
                '30x_Coverage_consensus_3of3_intersections', '30x_Coverage_consensus_3of3_unions',
                '30x_Coverage_cnvpytor', '30x_Coverage_gatk', '30x_Coverage_delly', 'SNP_Array'],
            "title": "Size Distributions for 30x Coverage Call Sets"
        },
        "6x": {
            "sets": ['6x_Coverage_consensus_1of3_intersections', '6x_Coverage_consensus_1of3_unions',
               '6x_Coverage_consensus_2of3_intersections', '6x_Coverage_consensus_2of3_unions',
               '6x_Coverage_consensus_3of3_intersections', '6x_Coverage_consensus_3of3_unions',
               '6x_Coverage_cnvpytor', '6x_Coverage_gatk', '6x_Coverage_delly', 'SNP_Array'],
            "title": "Size Distributions for 6x Coverage Call Sets"
        },
        "4x": {
            "sets": ['4x_Coverage_consensus_1of3_intersections', '4x_Coverage_consensus_1of3_unions',
                        '4x_Coverage_consensus_2of3_intersections', '4x_Coverage_consensus_2of3_unions',
                        '4x_Coverage_consensus_3of3_intersections', '4x_Coverage_consensus_3of3_unions',
                        '4x_Coverage_cnvpytor', '4x_Coverage_gatk', '4x_Coverage_delly', 'SNP_Array'],
            "title": "Size Distributions for 4x Coverage Call Sets"
        },
        "2x": {
            "sets": ['2x_Coverage_consensus_1of3_intersections', '2x_Coverage_consensus_1of3_unions',
                       '2x_Coverage_consensus_2of3_intersections', '2x_Coverage_consensus_2of3_unions',
                       '2x_Coverage_consensus_3of3_intersections', '2x_Coverage_consensus_3of3_unions',
                       '2x_Coverage_cnvpytor', '2x_Coverage_gatk', '2x_Coverage_delly', 'SNP_Array'],
            "title": "Size Distributions for 2x Coverage Call Sets"
        },
        "30x_no_unions": {
            "sets": ['30x_Coverage_consensus_1of3_intersections', '30x_Coverage_consensus_2of3_intersections',
                      '30x_Coverage_consensus_3of3_intersections', '30x_Coverage_cnvpytor', 
                      '30x_Coverage_gatk', '30x_Coverage_delly', 'SNP_Array'],
            "title": "Size Distributions for 30x Coverage Call Sets"
        },
        "6x_no_unions": {
            "sets": ['6x_Coverage_consensus_1of3_intersections', '6x_Coverage_consensus_2of3_intersections',
                      '6x_Coverage_consensus_3of3_intersections', '6x_Coverage_cnvpytor', 
                      '6x_Coverage_gatk', '6x_Coverage_delly', 'SNP_Array'],
            "title": "Size Distributions for 6x Coverage Call Sets"
        },
        "4x_no_unions": {
            "sets": ['4x_Coverage_consensus_1of3_intersections', '4x_Coverage_consensus_2of3_intersections',
                      '4x_Coverage_consensus_3of3_intersections', '4x_Coverage_cnvpytor', 
                      '4x_Coverage_gatk', '4x_Coverage_delly', 'SNP_Array'],
            "title": "Size Distributions for 4x Coverage Call Sets"
        },
        "2x_no_unions": {
            "sets": ['2x_Coverage_consensus_1of3_intersections', '2x_Coverage_consensus_2of3_intersections',
                      '2x_Coverage_consensus_3of3_intersections', '2x_Coverage_cnvpytor', 
                      '2x_Coverage_gatk', '2x_Coverage_delly', 'SNP_Array'],
            "title": "Size Distributions for 2x Coverage Call Sets"
        },
        "Intersections": {
            "sets": ['30x_Coverage_consensus_2of3_intersections', '6x_Coverage_consensus_2of3_intersections',
                      '4x_Coverage_consensus_2of3_intersections', '2x_Coverage_consensus_2of3_intersections', 'SNP_Array'],
            "title": "Size Distributions for Intersection Call Sets"
        },
        "Unions": {
            "sets": ['30x_Coverage_consensus_2of3_unions', '6x_Coverage_consensus_2of3_unions',
                      '4x_Coverage_consensus_2of3_unions', '2x_Coverage_consensus_2of3_unions', 'SNP_Array'],
            "title": "Size Distributions for Union Call Sets"
        },
        "CNVpytor": {
            "sets": ['30x_Coverage_cnvpytor', '6x_Coverage_cnvpytor', 
                     '4x_Coverage_cnvpytor', '2x_Coverage_cnvpytor', 'SNP_Array'],
            "title": "Size Distributions for CNVpytor Call Sets"
        },
        "GATK": {
            "sets": ['30x_Coverage_gatk', '6x_Coverage_gatk', 
                     '4x_Coverage_gatk', '2x_Coverage_gatk', 'SNP_Array'],
            "title": "Size Distributions for GATK Call Sets"
        },
        "DELLY": {
            "sets": ['30x_Coverage_delly', '6x_Coverage_delly', 
                     '4x_Coverage_delly', '2x_Coverage_delly', 'SNP_Array'],
            "title": "Size Distributions for DELLY Call Sets"
        },
    }

    # === Step 3-5: Generate All Plots in Parallel ===
    # Define all plotting tasks as partial functions for parallel execution
    plotting_tasks = [
        # Task 1: Statistical distributions
        # partial(
        #     plotter.plot_statistical_distributions,
        #     input_sets_to_plot=input_sets_to_plot,
        #     metrics=metrics,
        #     bounds=(500, 1_000_000),
        #     output_dir=output_dir / "figures" / "statistical_distributions",
        #     cumulative_stats_output_path=output_dir / "logs" / "statistical_distributions_cumulative_stats.tsv",
        # ),
        # # Task 1.5: Statistical distributions for all SV types only
        partial(
            plotter.plot_statistical_distributions,
            plot_config=statistical_distribution_input_sets,
            metrics=metrics,
            svtypes=[SVType.ALL],
            bounds=(500, 1_000_000),
            output_dir=output_dir / "figures" / "statistical_distributions_all_only",
            cumulative_stats_output_path=output_dir / "logs" / "statistical_distributions_cumulative_stats.tsv",
        ),
        # Task 2: Size distribution plots
        partial(
            plotter.plot_size_distribution,
            plot_config=size_distribution_input_sets,
            output_dir=output_dir / "figures" / "size_distributions",
            include_benchmark=True,
            stats_output_path=output_dir / "logs" / "size_distribution_stats.tsv",
        ),
        # Task 3: Caller source distribution
        partial(
            plotter.get_caller_source_distribution,
            input_sets_to_include=['30x_Coverage_consensus_2of3_intersections', '6x_Coverage_consensus_2of3_intersections',
                                   '4x_Coverage_consensus_2of3_intersections', '2x_Coverage_consensus_2of3_intersections'],
            output_file=output_dir / "figures" / "caller_source_distribution" / "caller_source_distribution.png",
        ),
        # Task 4: Get counts
        partial(
            plotter.get_count_statistics,
            output_file=output_dir / "logs" / "count_statistics.tsv",
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
    
    for input_name in ['30x Coverage', '6x Coverage', '4x Coverage', '2x Coverage']:
        plotting_tasks.append(
            partial(
                plotter.plot_count_venn_diagram,
                config=config,
                input_set_key=input_name,
                output_path=output_dir / "figures" / "venn_diagrams" / f"count_venn_diagram_{input_name.replace(' ', '_')}.png",
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
    config, args = parse_args()
    
    main(config)

