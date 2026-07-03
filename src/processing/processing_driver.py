from utils import parse_args, PipelineConfig
from processing.vcf_parser import process_vcfs_to_beds
from processing.processing_functions import run_consensus_calls_script


def main(config: PipelineConfig):
    print("\nStep 1: Converting VCF files to BED format...")
    _ = process_vcfs_to_beds(config, "input")

    print("\nStep 2: Running consensus calls script...")
    run_consensus_calls_script(config)


if __name__ == "__main__":
    # Allow running standalone for testing
    config = parse_args()
    main(config)
