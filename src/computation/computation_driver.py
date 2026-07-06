import json

from parsing_functions import parse_control_to_bed, parse_benchmarks_to_bed
from utils import parse_args, PipelineConfig
from computation.computation_functions import run_binary_classification_script

def main(config: PipelineConfig):
    """
    Main computation pipeline.

    Args:
        config: Parsed pipeline configuration
    """
    layout = config.layout

    print("\nStep 3: Processing control datasets (SNP Array)...")
    control_liftover_results = parse_control_to_bed(config)

    if control_liftover_results:
        liftover_log = layout.log("control_liftover_results.json")
        liftover_log.parent.mkdir(exist_ok=True, parents=True)
        with open(liftover_log, 'w') as f:
            json.dump(control_liftover_results, f, indent=4)

    print("\nStep 4: Parsing all benchmarks to BED format...")
    _, benchmark_liftover_results = parse_benchmarks_to_bed(config)

    if benchmark_liftover_results:
        liftover_log = layout.log("benchmark_liftover_results.json")
        liftover_log.parent.mkdir(exist_ok=True, parents=True)
        with open(liftover_log, 'w') as f:
            json.dump(benchmark_liftover_results, f, indent=4)

    print("\nStep 6: Running binary classification script...")
    bin_class_sets = []
    for key, input_map in config.input.items():
        # Add the consensus call sets (intersections, then unions) for classification
        for representation in ("intersections", "unions"):
            for level in (1, 2, 3):
                call_set = layout.consensus_call_set(level, representation)
                bin_class_sets.append((
                    str(layout.consensus_rep_dir(key, level, representation)),
                    str(layout.classification_dir(key, call_set)),
                ))

        # Add individual tool results to the sets for classification
        for tool in input_map.keys():
            bin_class_sets.append((
                str(layout.bed_tool_dir(key, tool)),
                str(layout.classification_dir(key, tool)),
            ))

    # Add control datasets to the sets for classification
    for key in config.control.keys():
        bin_class_sets.append((
            str(layout.control_bed_dir(key)),
            str(layout.control_classification_dir(key)),
        ))

    run_binary_classification_script(config, bin_class_sets)


if __name__ == "__main__":
    # Allow running standalone for testing
    config = parse_args()
    main(config)
