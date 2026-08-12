import pandas as pd

from consensuscnv.parsing.benchmark_parser import process_benchmarks_to_beds
from consensuscnv.parsing.exclusion_report import exclusion_summary, format_exclusion_summary
from consensuscnv.parsing.parser_utils import ExclusionMask
from consensuscnv.parsing.penncnv_parser import process_penncnv_to_beds
from consensuscnv.parsing.vcf_parser import process_vcfs_to_beds
from consensuscnv.utils import PipelineConfig


def parse_input_files(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parsing pipeline."""

    excluded_regions = ExclusionMask.load(config.excluded_regions_file)

    print("\nProcessing experimental datasets (VCF)...")
    vcf_statistics_list = process_vcfs_to_beds(config, excluded_regions)

    print("\nProcessing control datasets (SNP Array)...")
    penncnv_statistics = process_penncnv_to_beds(config, excluded_regions)

    print("\nParsing all benchmarks to BED format...")
    benchmark_statistics = process_benchmarks_to_beds(config, excluded_regions)

    # Return the statistics.
    vcf_statistics_df = pd.DataFrame(vcf_statistics_list)
    penncnv_statistics_df = pd.DataFrame.from_dict(penncnv_statistics, orient="index")
    benchmark_statistics_df = pd.DataFrame.from_dict(benchmark_statistics, orient="index")

    print(
        "\n"
        + format_exclusion_summary(
            exclusion_summary(
                vcf_statistics_df, penncnv_statistics_df, benchmark_statistics_df
            )
        )
    )

    return vcf_statistics_df, penncnv_statistics_df, benchmark_statistics_df
