"""Benchmark composition after the DEL/DUP isolation fix.

Where used
----------
Results -> "Input Call Sets After Parsing" -> the benchmark paragraph, and the
Table 1 benchmark rows.

Reports, per benchmark source and for the three sources merged into one truth
set: interval count, DEL/DUP split, median size, and the fraction of intervals
reaching 1 kb and 10 kb. The point of the table is the duplication column --
after insertions are dropped rather than folded into DUP, the merged truth set
retains almost no duplication content at CNV scale, which bounds what can be
claimed about duplication recall anywhere in the paper.

Merging uses the same settings as the truth side of the classification
(`max_padding=0`), so the merged row is the population the classifier actually
sees, not a naive concatenation.

    pixi run python manuscript/scripts/benchmark_composition.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from consensuscnv.callsets import collect_callsets, merge_components, read_bed_calls
from consensuscnv.callsets.registry import SVTYPES
from consensuscnv.classification.intervals import IntervalSet

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")
BENCHMARKS = ("1000G", "HGSVC3", "ont_vienna")
DEST = ROOT / "results" / "manuscript"

FLOORS = (1_000, 10_000)


def describe(label: str, sizes: np.ndarray, svtypes: np.ndarray) -> dict:
    row = {
        "call_set": label,
        "n_intervals": sizes.size,
        "n_del": int((svtypes == "DEL").sum()),
        "n_dup": int((svtypes == "DUP").sum()),
        "pct_del": 100.0 * (svtypes == "DEL").mean(),
        "median_size": float(np.median(sizes)),
    }
    for floor in FLOORS:
        row[f"pct_ge_{floor}"] = 100.0 * (sizes >= floor).mean()
        row[f"n_dup_ge_{floor}"] = int(((svtypes == "DUP") & (sizes >= floor)).sum())
    return row


def as_interval_set(paths) -> IntervalSet:
    calls = collect_callsets(read_bed_calls(p) for p in paths)
    return IntervalSet.from_merged(merge_components(calls, max_padding=0))


rows = []
all_paths = []
for name in BENCHMARKS:
    paths = sorted((ROOT / "out" / "benchmark" / name).glob("*.bed"))
    all_paths.extend(paths)
    intervals = as_interval_set(paths)
    rows.append(
        describe(
            name,
            intervals.lengths,
            np.array(SVTYPES.names)[intervals.svtype_idx],
        )
    )

merged = as_interval_set(all_paths)
rows.append(
    describe(
        "merged truth set",
        merged.lengths,
        np.array(SVTYPES.names)[merged.svtype_idx],
    )
)

table = pd.DataFrame(rows)
DEST.mkdir(parents=True, exist_ok=True)
table.to_csv(DEST / "benchmark_composition.csv", index=False)

pd.set_option("display.width", 220)
print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
print(f"\nwrote {DEST / 'benchmark_composition.csv'}")
