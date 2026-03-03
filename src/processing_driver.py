import yaml
import time
from pathlib import Path

from utils import parse_args
from computation_functions import convert_vcfs_to_bed, run_consensus_calls_script, convert_control_to_bed, perform_liftover

def main(config: dict, debug: bool = False):
    """
    Main computation pipeline.
    
    Args:
        config: Configuration dictionary loaded from YAML
    """

    time_0 = time.time()

    print("\nStep 1: Converting VCF files to BED format...")
    convert_vcfs_to_bed(config)

    time_1 = time.time()

    print("\nStep 2: Running consensus calls script...")
    run_consensus_calls_script(config)

    print("\nStep 3: Processing control datasets (SNP Array)...")
    convert_control_to_bed(config)

    time_1 = time.time()

    print("\nStep 4: Performing liftover on datasets (if configured)...")
    liftover_log = Path(config['output_dir']) / "logs" / "liftover_results.json"
    perform_liftover(config, log_file=liftover_log)

    time_2 = time.time()

    if debug:
        print("\n=== DEBUG TIMING INFO ===")
        print(f"VCF to BED conversion time: {time_1 - time_0:.2f} seconds")
        print(f"Consensus calls script time: {time_2 - time_1:.2f} seconds")
        print(f"Total consensus pipeline time: {time_2 - time_0:.2f} seconds")

    

if __name__ == "__main__":
    # Allow running standalone for testing
    config = parse_args()
    
    main(config)
