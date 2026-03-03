import yaml
from pathlib import Path
import time

from utils import parse_args
from computation_functions import convert_control_to_bed, perform_liftover, run_benchmark_processing_script, run_binary_classification_script

def main(config: dict, debug: bool = False):
    """
    Main computation pipeline.
    
    Args:
        config: Configuration dictionary loaded from YAML
    """

    time_0 = time.time()

    print("\nStep 5: Running benchmark processing script...")
    benchmark_log = Path(config['output_dir']) / "logs" / "benchmark_processing_results.json"
    run_benchmark_processing_script(config, log_file=benchmark_log)

    time_1 = time.time()

    print("\nStep 6: Running binary classification script...")
    run_binary_classification_script(config)

    time_2 = time.time()

    if debug:
        print("\n=== DEBUG TIMING INFO ===")
        print(f"Benchmark processing time: {time_1 - time_0:.2f} seconds")
        print(f"Binary classification processing time: {time_2 - time_1:.2f} seconds")
        print(f"Total computation pipeline time: {time_2 - time_0:.2f} seconds")

    

if __name__ == "__main__":
    # Allow running standalone for testing
    config = parse_args()
    
    main(config)