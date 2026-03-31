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
    run_binary_classification_script(config)


if __name__ == "__main__":
    # Allow running standalone for testing
    config = parse_args()
    main(config)