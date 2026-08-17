"""Table 1 (main text) and Supplementary Tables S1-S2: parsing summary.

Where used
----------
Results -> "Input Call Sets After Parsing" -> Table 1.
    Post-exclusion, analysis-ready counts for every parsed call set: the three
    sequence-based callers at each coverage, the SNP array control, and each
    benchmark source. Carries one QC column, `% calls removed by mask`, because
    that quantity varies from <0.1% to >45% across arms and is itself a result.

Supplementary Table S1: pre-exclusion counts and the full liftover accounting.
Supplementary Table S2: full exclusion accounting (bases removed vs. masked,
    the collateral ratio, and the per-SVTYPE split).

Inputs are the parsed BEDs under `out/` plus the statistics CSVs written by
`src/test_parsing.py` into `results/parsing/`. Run after parsing:

    pixi run python manuscript/scripts/table1_parsing_summary.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")
OUT = ROOT / "out"
PARSING = ROOT / "results" / "parsing"
DEST = ROOT / "results" / "manuscript"

COVERAGES = ("30x", "6x", "4x", "2x")
CALLERS = ("cnvpytor", "delly", "gatk")
BENCHMARKS = ("1000G", "HGSVC3", "ont_vienna")

# (label, directory, key into exclusion_summary as (dataset, source))
CALL_SETS = (
    [
        (f"{cov} {caller}", OUT / f"{cov}_Coverage" / caller, (f"{cov} Coverage", caller))
        for cov in COVERAGES
        for caller in CALLERS
    ]
    + [("SNP Array", OUT / "SNP_Array" / "bed", ("SNP Array", "SNP Array"))]
    + [(b, OUT / "benchmark" / b, (b, b)) for b in BENCHMARKS]
)


def read_beds(directory: Path) -> pd.DataFrame:
    """Every per-sample BED in a directory as one frame, sample id from the filename."""
    frames = []
    for path in sorted(directory.glob("*.bed")):
        frame = pd.read_csv(
            path, sep="\t", header=None,
            names=["chrom", "start", "end", "svtype", "source"],
            dtype={"chrom": str},
        )
        frame["sample_id"] = path.stem
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise(label: str, calls: pd.DataFrame, removed_pct: float) -> dict:
    per_sample = calls.groupby("sample_id").size()
    size = (calls["end"] - calls["start"]).to_numpy()
    median_calls = float(per_sample.median())
    return {
        "call_set": label,
        "n_samples": calls["sample_id"].nunique(),
        "n_calls": len(calls),
        "median_per_sample": median_calls,
        "mad_per_sample": float(np.median(np.abs(per_sample - median_calls))),
        "pct_del": 100.0 * (calls["svtype"] == "DEL").mean(),
        "median_size": float(np.median(size)),
        "iqr_size": float(np.percentile(size, 75) - np.percentile(size, 25)),
        "pct_removed_by_mask": removed_pct,
    }


exclusions = pd.read_csv(PARSING / "exclusion_summary.csv")
removed = {
    (row.dataset, row.source): 100.0 * row.pct_removed
    for row in exclusions.itertuples()
}

table1 = pd.DataFrame(
    [summarise(label, read_beds(d), removed.get(key, float("nan")))
     for label, d, key in CALL_SETS]
)

# --- Supplementary S1: pre-exclusion counts + liftover ------------------------
pre = exclusions[["input_type", "dataset", "source", "n_calls", "n_removed"]].copy()
pre = pre.rename(columns={"n_calls": "n_calls_pre_mask", "n_removed": "n_calls_removed"})
pre["n_calls_post_mask"] = pre["n_calls_pre_mask"] - pre["n_calls_removed"]

# --- Supplementary S2: full exclusion accounting ------------------------------
supp2 = exclusions[[
    "input_type", "dataset", "source", "n_calls", "n_removed", "pct_removed",
    "pct_del_removed", "pct_dup_removed", "mb_removed", "mb_masked",
    "pct_bases_removed", "collateral",
]]

DEST.mkdir(parents=True, exist_ok=True)
table1.to_csv(DEST / "table1_parsing_summary.csv", index=False)
pre.to_csv(DEST / "supp_s1_pre_exclusion.csv", index=False)
supp2.to_csv(DEST / "supp_s2_exclusion_detail.csv", index=False)

pd.set_option("display.width", 200)
print("=== Table 1: parsed call sets, post-exclusion ===")
print(table1.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
print("\n=== Supplementary S1: pre- vs post-mask ===")
print(pre.to_string(index=False))
print(f"\nwrote 3 CSVs to {DEST}")
