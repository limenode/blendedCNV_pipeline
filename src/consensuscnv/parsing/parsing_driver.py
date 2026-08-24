import pandas as pd

from consensuscnv.parsing.benchmark_parser import process_benchmarks_to_beds
from consensuscnv.parsing.exclusion_report import exclusion_summary, format_exclusion_summary
from consensuscnv.parsing.parser_utils import ExclusionMask, load_sample_list
from consensuscnv.parsing.penncnv_parser import process_penncnv_to_beds
from consensuscnv.parsing.vcf_parser import process_vcfs_to_beds
from consensuscnv.utils import PipelineConfig


def parse_input_files(config: PipelineConfig, max_excluded_fraction: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs parsers for experimental control, and benchmark datasets.
    Does not parse any records that fall within an interval in the exclusion mask by
    at least {max_excluded_fraction} percentage of a call's length.
    """

    excluded_regions = ExclusionMask.load(config.excluded_regions_file)

    samples = load_sample_list(config.sample_list_file)

    print("\nProcessing experimental datasets...")
    vcf_statistics_list = process_vcfs_to_beds(
        config, excluded_regions, max_excluded_fraction=max_excluded_fraction, samples=samples
    )

    print("\nProcessing control datasets...")
    penncnv_statistics = process_penncnv_to_beds(
        config, excluded_regions, max_excluded_fraction=max_excluded_fraction, samples=samples
    )

    print("\nProcessing benchmark datasets...")
    benchmark_statistics = process_benchmarks_to_beds(
        config, excluded_regions, max_excluded_fraction=max_excluded_fraction, samples=samples
    )

    # Parse statistics to dataframes
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
