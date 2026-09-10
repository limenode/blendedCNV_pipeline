"""Parse every input call set and record the parsing and exclusion statistics.

Where used
----------
Results -> "Input Call Sets After Parsing":
    Table 1     via `table1_parsing_summary.py`, which reads
                `results/parsing/exclusion_summary.csv` written here
    Supplementary Table S1  pre-exclusion counts and the liftover accounting
    Supplementary Table S2  the full exclusion accounting

This is the parsing step the paper's numbers rest on, kept as a script because
`table1_parsing_summary.py` needs the per-parser statistics frames and the CLI
does not write them: `consensuscnv call` and `consensuscnv benchmark` write BED
files under `output_dir`, not statistics under `results/`.

Run this before `table1_parsing_summary.py`. It rewrites every BED under
`out/`, so the consensus and evaluation steps should be re-run after it.

    pixi run python manuscript/scripts/parse_inputs.py
"""

from pathlib import Path

from consensuscnv.parsing.exclusion_report import exclusion_summary
from consensuscnv.parsing.parsing_driver import parse_input_files
from consensuscnv.utils import build_config

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")
CONFIG = ROOT / "internal.config.yaml"
DEST = ROOT / "results" / "parsing"

config = build_config(CONFIG)

vcf_statistics, penncnv_statistics, benchmark_statistics = parse_input_files(config)

DEST.mkdir(parents=True, exist_ok=True)
vcf_statistics.to_csv(DEST / "vcf_statistics.csv", index=False)
penncnv_statistics.to_csv(DEST / "penncnv_statistics.csv")
benchmark_statistics.to_csv(DEST / "benchmark_statistics.csv")

# One row per (input type, dataset, source), normalised across all three parsers.
# This is the file `table1_parsing_summary.py` reads.
exclusions = exclusion_summary(vcf_statistics, penncnv_statistics, benchmark_statistics)
exclusions.to_csv(DEST / "exclusion_summary.csv", index=False)

print(f"\nWrote parsing statistics to {DEST}")
