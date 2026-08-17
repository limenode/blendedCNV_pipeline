# %% Imports and constants

from pathlib import Path
import os

from consensuscnv.parsing.exclusion_report import (
    exclusion_summary
)
from consensuscnv.parsing.parsing_driver import parse_input_files
from consensuscnv.utils import build_config

RESULTS_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/results")

# %% Build config
config = build_config(
    Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/internal.config.yaml")
)

# %% Run parsing and get statistics
vcf_statistics, penncnv_statistics, benchmark_statistics = parse_input_files(config, max_excluded_fraction=0.01)

# %% Output statistics to CSV files
os.makedirs(RESULTS_DIR / "parsing", exist_ok=True)
vcf_statistics.to_csv(RESULTS_DIR / "parsing" / "vcf_statistics.csv", index=False)
penncnv_statistics.to_csv(RESULTS_DIR / "parsing" / "penncnv_statistics.csv")
benchmark_statistics.to_csv(RESULTS_DIR / "parsing" / "benchmark_statistics.csv")

# %% Exclusion mask summary
# One row per (input type, dataset, source), normalised across all three parsers.
exclusions = exclusion_summary(vcf_statistics, penncnv_statistics, benchmark_statistics)
exclusions.to_csv(RESULTS_DIR / "parsing" / "exclusion_summary.csv", index=False)
