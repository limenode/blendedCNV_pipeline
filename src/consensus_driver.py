import yaml
import time

from utils import parse_args
from computation_functions import convert_vcfs_to_bed, run_consensus_calls_script

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

    time_2 = time.time()

    if debug:
        print("\n=== DEBUG TIMING INFO ===")
        print(f"VCF to BED conversion time: {time_1 - time_0:.2f} seconds")
        print(f"Consensus calls script time: {time_2 - time_1:.2f} seconds")
        print(f"Total consensus pipeline time: {time_2 - time_0:.2f} seconds")

    

if __name__ == "__main__":
    # Allow running standalone for testing
    args = parse_args()
    
    # Load configuration from YAML file
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)