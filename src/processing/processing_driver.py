import json
from pathlib import Path

from utils import parse_args
from parsing_functions import parse_vcfs_to_bed
from processing.processing_functions import run_consensus_calls_script

def main(config: dict):

    print("\nStep 1: Converting VCF files to BED format...")
    input_liftover_results = parse_vcfs_to_bed(config)

    if input_liftover_results:
        liftover_log = Path(config['output_dir']) / "logs" / "input_liftover_results.json"
        liftover_log.parent.mkdir(exist_ok=True, parents=True)
        with open(liftover_log, 'w') as f:
            json.dump(input_liftover_results, f, indent=4)

    print("\nStep 2: Running consensus calls script...")
    run_consensus_calls_script(config)
    

if __name__ == "__main__":
    # Allow running standalone for testing
    config, args = parse_args()
    main(config)
