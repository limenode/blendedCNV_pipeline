import yaml
import time
import json
from pathlib import Path

from utils import parse_args
from new_functions import convert_vcfs_to_bed, convert_control_to_bed
from processing_functions import run_consensus_calls_script

def main(config: dict, debug: bool = False):
    """
    Main computation pipeline.
    
    Args:
        config: Configuration dictionary loaded from YAML
    """

    time_0 = time.time()

    print("\nStep 1: Converting VCF files to BED format...")
    input_liftover_results = convert_vcfs_to_bed(config)

    time_1 = time.time()

    print("\nStep 2: Running consensus calls script...")
    run_consensus_calls_script(config)

    time_2 = time.time()

    print("\nStep 3: Processing control datasets (SNP Array)...")
    control_liftover_results = convert_control_to_bed(config)

    time_2 = time.time()

    liftover_results = {
        'input': input_liftover_results,
        'control': control_liftover_results
    }

    if liftover_results:
        # Save results to log file
        liftover_log = Path(config['output_dir']) / "logs" / "liftover_results.json"
        liftover_log.parent.mkdir(exist_ok=True, parents=True)
        with open(liftover_log, 'w') as f:
            json.dump(liftover_results, f, indent=4)

    time_3 = time.time()

    if debug:
        print("\n=== DEBUG TIMING INFO ===")
        print(f"VCF to BED conversion time: {time_1 - time_0:.2f} seconds")
        print(f"Consensus calls processing time: {time_2 - time_1:.2f} seconds")
        print(f"Control dataset processing time: {time_2 - time_1:.2f} seconds")
        print(f"Dataset liftover time: {time_3 - time_2:.2f} seconds")
        print(f"Total preprocessing pipeline time: {time_3 - time_0:.2f} seconds")

    

if __name__ == "__main__":
    # Allow running standalone for testing
    config = parse_args()
    
    main(config)
