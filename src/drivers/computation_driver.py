from pathlib import Path
import json

from parsing_functions import parse_control_to_bed, parse_benchmarks_to_bed
from utils import parse_args
from computation_functions import run_binary_classification_script

def main(config: dict):
    """
    Main computation pipeline.
    
    Args:
        config: Configuration dictionary loaded from YAML
    """

    print("\nStep 3: Processing control datasets (SNP Array)...")
    control_liftover_results = parse_control_to_bed(config)

    if control_liftover_results:
        liftover_log = Path(config['output_dir']) / "logs" / "control_liftover_results.json"
        liftover_log.parent.mkdir(exist_ok=True, parents=True)
        with open(liftover_log, 'w') as f:
            json.dump(control_liftover_results, f, indent=4)

    print("\nStep 4: Parsing all benchmarks to BED format...")
    _, benchmark_liftover_results = parse_benchmarks_to_bed(config)

    if benchmark_liftover_results:
        liftover_log = Path(config['output_dir']) / "logs" / "benchmark_liftover_results.json"
        liftover_log.parent.mkdir(exist_ok=True, parents=True)
        with open(liftover_log, 'w') as f:
            json.dump(benchmark_liftover_results, f, indent=4)

    print("\nStep 6: Running binary classification script...")
    bin_class_sets = []
    for key, input_map in config['input'].items():
        set_subdir_name = key.replace(" ", "_")
        set_subdir = Path(config['output_dir']) / set_subdir_name
        
        # Add the intersections path to the sets for classification
        bin_class_sets.append((str(set_subdir / "consensus_1of3" / "intersections"), str(set_subdir / "binary_classification" / "consensus_1of3_intersections")))
        bin_class_sets.append((str(set_subdir / "consensus_2of3" / "intersections"), str(set_subdir / "binary_classification" / "consensus_2of3_intersections")))
        bin_class_sets.append((str(set_subdir / "consensus_3of3" / "intersections"), str(set_subdir / "binary_classification" / "consensus_3of3_intersections")))

        # Add the unions path to the sets for classification
        bin_class_sets.append((str(set_subdir / "consensus_1of3" / "unions"), str(set_subdir / "binary_classification" / "consensus_1of3_unions")))
        bin_class_sets.append((str(set_subdir / "consensus_2of3" / "unions"), str(set_subdir / "binary_classification" / "consensus_2of3_unions")))
        bin_class_sets.append((str(set_subdir / "consensus_3of3" / "unions"), str(set_subdir / "binary_classification" / "consensus_3of3_unions")))

        # Add individual tool results to the sets for classification
        for tool in input_map.keys():
            bin_class_sets.append((str(set_subdir / "bed" / tool), str(set_subdir / "binary_classification" / tool)))
        
    # Add control datasets to the sets for classification
    for key, _ in config['control'].items():
        control_subdir_name = key.replace(" ", "_")
        control_subdir = Path(config['output_dir']) / control_subdir_name
        bin_class_sets.append((str(control_subdir / "bed"), str(control_subdir / "binary_classification")))

    run_binary_classification_script(config, bin_class_sets)


if __name__ == "__main__":
    # Allow running standalone for testing
    config, args = parse_args()
    main(config)
