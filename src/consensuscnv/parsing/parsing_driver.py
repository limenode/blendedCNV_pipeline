from consensuscnv.parsing.benchmark_parser import process_benchmarks_to_beds
from consensuscnv.parsing.parser_utils import ExclusionMask
from consensuscnv.parsing.penncnv_parser import process_penncnv_to_beds
from consensuscnv.parsing.vcf_parser import process_vcfs_to_beds
from consensuscnv.utils import PipelineConfig


def parse_input_files(config: PipelineConfig) -> None:
    """Parsing pipeline."""

    excluded_regions = ExclusionMask.load(config.excluded_regions_file)

    print("\nProcessing experimental datasets (VCF)...")
    _ = process_vcfs_to_beds(config, excluded_regions)

    print("\nProcessing control datasets (SNP Array)...")
    _ = process_penncnv_to_beds(config, excluded_regions)

    print("\nParsing all benchmarks to BED format...")
    _ = process_benchmarks_to_beds(config, excluded_regions)
