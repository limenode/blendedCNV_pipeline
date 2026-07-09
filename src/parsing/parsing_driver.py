from utils import PipelineConfig
from parsing.vcf_parser import process_vcfs_to_beds
from parsing.benchmark_parser import process_benchmarks_to_beds
from parsing.penncnv_parser import process_penncnv_to_beds

def parse_input_files(config: PipelineConfig):
    """Parsing pipeline."""
    
    print("\nProcessing experimental datasets (VCF)...")
    _ = process_vcfs_to_beds(config)
    
    print("\nProcessing control datasets (SNP Array)...")
    _ = process_penncnv_to_beds(config)

    print("\nParsing all benchmarks to BED format...")
    _ = process_benchmarks_to_beds(config)

