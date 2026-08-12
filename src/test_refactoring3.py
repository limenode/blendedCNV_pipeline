# %% Imports and constants

from pathlib import Path
import os

from consensuscnv.parsing.exclusion_report import (
    exclusion_summary,
    format_exclusion_summary,
)
from consensuscnv.parsing.parsing_driver import parse_input_files
from consensuscnv.utils import build_config

BENCHMARK_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark")
QUERY_DIR: Path = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/30x_Coverage")
TEST_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/src/test")

# %% Build config
config = build_config(
    Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/internal.config.yaml")
)

# %% Run parsing and get statistics
vcf_statistics, penncnv_statistics, benchmark_statistics = parse_input_files(config)

# %% Output statistics to CSV files
os.makedirs(TEST_DIR / "parsing", exist_ok=True)
vcf_statistics.to_csv(TEST_DIR / "parsing" / "vcf_statistics.csv", index=False)
penncnv_statistics.to_csv(TEST_DIR / "parsing" / "penncnv_statistics.csv")
benchmark_statistics.to_csv(TEST_DIR / "parsing" / "benchmark_statistics.csv")

# %% Exclusion mask summary
# One row per (input type, dataset, source), normalised across all three parsers.
exclusions = exclusion_summary(vcf_statistics, penncnv_statistics, benchmark_statistics)
exclusions.to_csv(TEST_DIR / "parsing" / "exclusion_summary.csv", index=False)

print(format_exclusion_summary(exclusions))          # compact view
