from utils import parse_args, PipelineConfig
from processing.vcf_parser import process_vcfs_to_beds
from processing.consensus_calling import compute_consensus_from_beds

def main(config: PipelineConfig):
    print("\nStep 1: Converting VCF files to BED format...")
    _ = process_vcfs_to_beds(config, "input")

    print("\nStep 2: Running consensus calls script...")
    compute_consensus_from_beds(config, weight_threshold=0.5)

if __name__ == "__main__":
    # Allow running standalone for testing
    config = parse_args()
    main(config)
