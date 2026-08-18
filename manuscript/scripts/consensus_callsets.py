"""Consensus call set composition across coverages and agreement levels.

Where used
----------
Results -> "Consensus Call Set Construction": the caller-agreement counts behind
Figure 2, and every number in the prose describing how consensus call sets
respond to coverage and to the agreement requirement.

Reports, for each coverage and each agreement level (1/3, 2/3, 3/3), the number
of consensus components, the percentage that are duplications, and the median
size; and, for 30x, the seven caller-agreement regions of Figure 2.

Consensus is a union merge at 50% reciprocal overlap with no padding. All three
levels come off a single merge, since `merge_components` builds components from
the edge selection alone and drops those below `min_sources` only afterwards.

    pixi run python manuscript/scripts/consensus_callsets.py
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from consensuscnv.callsets import collect_callsets, merge_components, read_bed_calls
from consensuscnv.callsets.registry import SOURCES, SVTYPES
from consensuscnv.classification.intervals import IntervalSet

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")
DEST = ROOT / "results" / "manuscript"

COVERAGES = ("30x", "6x", "4x", "2x")
CALLERS = ("cnvpytor", "delly", "gatk")
CONSENSUS_THRESHOLD = 0.5
LEVELS = (1, 2, 3)


def beds(coverage: str, subdirs: tuple[str, ...]) -> list[str]:
    root = ROOT / "out" / f"{coverage}_Coverage"
    return [b for sub in subdirs for b in sorted(glob.glob(str(root / sub / "*.bed")))]


def describe(coverage: str, label: str, intervals: IntervalSet) -> dict:
    svtypes = np.array(SVTYPES.names)[intervals.svtype_idx]
    return {
        "coverage": coverage,
        "call_set": label,
        "n_calls": len(intervals),
        "pct_dup": 100.0 * (svtypes == "DUP").mean(),
        "median_size": float(np.median(intervals.lengths)),
    }


rows = []
for coverage in COVERAGES:
    # Raw per-caller sets, read from their own directories with no merging, so
    # the extents are the caller's own rather than a component's.
    for caller in CALLERS:
        raw = IntervalSet.from_callset(
            collect_callsets(read_bed_calls(b) for b in beds(coverage, (caller,)))
        )
        rows.append(describe(coverage, caller, raw))

    merged = IntervalSet.from_merged(
        merge_components(
            collect_callsets(read_bed_calls(b) for b in beds(coverage, CALLERS)),
            min_reciprocal_overlap=CONSENSUS_THRESHOLD,
        )
    )
    for level in LEVELS:
        rows.append(
            describe(coverage, f"{level}/3", merged.select(merged.n_sources >= level))
        )

    if coverage == "30x":
        # source_bits already is the set of callers behind a component, so the
        # seven Figure 2 regions are a bincount over the masks. SOURCES ids are
        # assigned in first-appearance order, so each mask is rebuilt from the
        # key rather than assumed to be 1/2/4.
        caller_bit = {c: 1 << SOURCES.get(c) for c in CALLERS}
        caller_mask = sum(caller_bit.values())
        assert not (merged.source_bits & ~caller_mask).any(), "non-caller source present"
        counts = np.bincount(merged.source_bits, minlength=caller_mask + 1)
        venn = pd.DataFrame(
            [
                {
                    "callers": "|".join(c for c, on in zip(CALLERS, key) if on == "1"),
                    "n_callers": key.count("1"),
                    "n_calls": int(
                        counts[sum(caller_bit[c] for c, on in zip(CALLERS, key) if on == "1")]
                    ),
                }
                for key in ("100", "010", "110", "001", "101", "011", "111")
            ]
        )
        venn["pct_of_1_of_3"] = 100.0 * venn["n_calls"] / len(merged)
        assert venn["n_calls"].sum() == len(merged)

table = pd.DataFrame(rows)
wide = table.pivot(index="call_set", columns="coverage", values="n_calls")[list(COVERAGES)]

DEST.mkdir(parents=True, exist_ok=True)
table.to_csv(DEST / "consensus_callsets.csv", index=False)
venn.to_csv(DEST / "consensus_caller_agreement_30x.csv", index=False)

pd.set_option("display.width", 200)
print("=== per coverage and call set ===")
print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
print("\n=== call counts, wide ===")
print(wide.to_string())
print("\n=== Figure 2 regions, 30x ===")
print(venn.sort_values(["n_callers", "n_calls"], ascending=[True, False]).to_string(
    index=False, float_format=lambda v: f"{v:.2f}"))

print("\n=== change in call count with the agreement requirement ===")
for coverage in COVERAGES:
    counts = {r["call_set"]: r["n_calls"] for r in rows if r["coverage"] == coverage}
    drop_12 = 100 * (1 - counts["2/3"] / counts["1/3"])
    drop_23 = 100 * (1 - counts["3/3"] / counts["2/3"])
    print(f"  {coverage:>4}: 1/3 {counts['1/3']:>6,} -> 2/3 {counts['2/3']:>6,} "
          f"({drop_12:5.1f}% fewer) -> 3/3 {counts['3/3']:>6,} ({drop_23:5.1f}% fewer)")

print("\n=== duplication share by level ===")
for coverage in COVERAGES:
    shares = {r["call_set"]: r["pct_dup"] for r in rows if r["coverage"] == coverage}
    print(f"  {coverage:>4}: " + "  ".join(f"{k} {shares[k]:5.1f}%" for k in ("1/3", "2/3", "3/3")))


# --- Supplementary Table Callsets, rendered as a Typst table ---------------- #
# Emitted as markup rather than as a CSV so the supplementary table is generated
# from the same numbers as the main text and never diverges from them by hand.
LABELS = {"cnvpytor": "CNVpytor", "delly": "Delly", "gatk": "GATK-gCNV"}
ORDER = ("cnvpytor", "delly", "gatk", "1/3", "2/3", "3/3")

lines = [
    "#block(width: 100%)[",
    "#set text(hyphenate: false, size: 10pt)",
    "#table(",
    "  columns: (auto, 1fr, auto, auto, auto),",
    "  align: (left, left, right, right, right),",
    "  table.header(",
    "    [Coverage], [Call Set], [Calls], [% DUP], [Median\\ Size (bp)],",
    "  ),",
]
by_key = {(r["coverage"], r["call_set"]): r for r in rows}
for position, coverage in enumerate(COVERAGES):
    if position:
        lines.append("  table.hline(stroke: 0.3pt),")
    for index, name in enumerate(ORDER):
        record = by_key[(coverage, name)]
        label = LABELS.get(name, f"{name} consensus")
        lines.append(
            f"  [{f'*{coverage}*' if index == 0 else ''}], [{label}], "
            f"[{record['n_calls']:,}], [{record['pct_dup']:.2f}], "
            f"[{record['median_size']:,.0f}],"
        )
lines += [")", "]"]

(DEST / "supp_callsets_table.typ").write_text("\n".join(lines) + "\n")
print(f"\nwrote {DEST / 'supp_callsets_table.typ'}")
