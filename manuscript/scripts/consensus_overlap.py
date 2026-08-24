"""Query consensus reciprocal-overlap sweep behind the choice of 0.5.

Where used
----------
Results -> "Parameterizing the Comparison" -> consensus reciprocal overlap:
    Figure (consensus overlap)  the four-panel sweep
    every number quoted in that subsection

This is the only parameter that changes what a *query call is*. A consensus
component is the transitive closure of the pairwise overlaps that clear the
threshold, so lowering it does not merely merge more pairs: it chains. Two calls
that share no bases at all end up in one component whenever a third bridges
them, and the component inherits the union of every member's span.

The direct measure of that is carried here as *span inflation*: the ratio of a
component's span to the span of its longest single member call. A component
whose ratio is 1 is a set of calls agreeing on one event; a ratio of 10 is a
locus collapsed into one interval that no caller reported.

Truth is held fixed throughout (benchmark merged at zero padding, 1 kb floor),
so every recall denominator in the sweep is the same number and the curves are
a property of the query side alone.

    pixi run python manuscript/scripts/consensus_overlap.py
"""

import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from consensuscnv.callsets import collect_callsets, merge_components, read_bed_calls
from consensuscnv.classification.classify import classify
from consensuscnv.classification.intervals import IntervalSet
from consensuscnv.classification.pairs import build_candidates

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")
DEST = ROOT / "results" / "parameterization"
TABLES = ROOT / "results" / "manuscript"

CALLERS = ("cnvpytor", "delly", "gatk")
BENCHMARKS = ("1000G", "HGSVC3", "ont_vienna")
CONSENSUS_LEVELS = (1, 2, 3)

SIZE_FLOOR = 1_000          # the detectable size domain, fixed upstream
BENCHMARK_PADDING = 0       # bridges exactly-touching records, nothing wider
CLASSIFY_THRESHOLD = 0.5    # held while the consensus threshold is swept
ADOPTED = 0.5               # the value this sweep is asked to justify

# Consensus level is ordinal, so the three curves take a sequential ramp
# (ColorBrewer Purples) rather than three unrelated hues. Same assignment as the
# size-domain and padding figures, so a call set keeps one identity across the
# paper.
LEVEL_COLORS = {"1/3": "#9E9AC8", "2/3": "#6A51A3", "3/3": "#3F007D"}
LEVELS = ["1/3", "2/3", "3/3"]


def bed_paths(root: Path, subdirs: tuple[str, ...]) -> list[str]:
    """Every per-sample BED under the named subdirectories of `root`.

    Pinned to an explicit tuple rather than a wildcard: consensus output is
    destined to land beside the per-caller directories, and a glob would read it
    back in as a fourth source.
    """
    return [bed for sub in subdirs for bed in sorted(glob.glob(str(root / sub / "*.bed")))]


# --------------------------------------------------------------------------- #
# Call sets
# --------------------------------------------------------------------------- #
truth = IntervalSet.from_merged(
    merge_components(
        collect_callsets(read_bed_calls(bed) for bed in bed_paths(ROOT / "out" / "benchmark", BENCHMARKS)),
        max_padding=BENCHMARK_PADDING,
    )
).filter_by_size(min_size=SIZE_FLOOR)

# Built once. The overlap edge list is recorded in full and sorted by reciprocal
# overlap, so each threshold in the sweep is a searchsorted and a slice off this
# object; only the connected-component pass is repeated.
raw = collect_callsets(read_bed_calls(bed) for bed in bed_paths(ROOT / "out" / "30x_Coverage", CALLERS))
member_lengths = raw.ends - raw.starts

thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
rows: list[dict[str, float]] = []

for threshold in thresholds:
    merged = merge_components(raw, min_reciprocal_overlap=float(threshold))
    # Span of the longest single call inside each component, so a component's
    # span can be read against the largest thing any one caller actually
    # reported there.
    longest = np.zeros(int(merged.labels.max()) + 1, dtype=np.int64)
    np.maximum.at(longest, merged.labels, member_lengths)
    longest = longest[merged.component_id]

    intervals = IntervalSet.from_merged(merged)
    lengths, n_sources = intervals.lengths, intervals.n_sources

    for level in CONSENSUS_LEVELS:
        # min_sources is a post-hoc component filter, so selecting n_sources >= k
        # off the single unfiltered merge is exactly what min_sources=k returns.
        keep = np.flatnonzero((n_sources >= level) & (lengths >= SIZE_FLOOR))
        query = intervals.select(keep)
        summary = classify(
            build_candidates(query, truth),
            min_reciprocal_overlap=CLASSIFY_THRESHOLD,
            validate=False,
        ).summary()
        inflation = lengths[keep] / longest[keep]
        rows.append({
            "consensus_threshold": float(threshold),
            "call_set": f"{level}/3",
            "n_query_all": int((n_sources >= level).sum()),
            "n_query": summary.n_query,
            "n_truth": summary.n_truth,
            "precision": summary.precision,
            "recall": summary.recall,
            "f1": summary.f1,
            "n_true_positive": summary.n_true_positive,
            "n_truth_found": summary.n_truth_found,
            "median_size": float(np.median(lengths[keep])),
            "median_inflation": float(np.median(inflation)),
            "p95_inflation": float(np.percentile(inflation, 95)),
            "max_inflation": float(inflation.max()),
            "median_n_calls": float(np.median(merged.n_calls[keep])),
        })

curves = pd.DataFrame(rows).set_index(["call_set", "consensus_threshold"]).sort_index()

# The truth side is untouched by this parameter, so every recall denominator in
# the sweep has to be the one number. If it is not, the comparison below is
# between moving targets.
assert curves["n_truth"].nunique() == 1, "truth set moved during the consensus sweep"

TABLES.mkdir(parents=True, exist_ok=True)
curves.to_csv(TABLES / "consensus_overlap_curves.csv")


# --------------------------------------------------------------------------- #
# Shared style
# --------------------------------------------------------------------------- #
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#000000",
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 6.5,
    "lines.solid_capstyle": "round",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def dress(ax, ylabel: str, letter: str, bottom: bool) -> None:
    ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel, labelpad=2)
    # The adopted value, drawn light and dotted on every panel so the same
    # vertical reference reads across all four without competing with data.
    ax.axvline(ADOPTED, color="#B0B0B0", linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="x", labelbottom=True, length=3.4)
    ax.tick_params(axis="y", length=3.4)
    if bottom:
        ax.set_xlabel("Consensus reciprocal-overlap threshold", labelpad=2)
    ax.text(-0.155, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def series(ax, column, **kwargs):
    for level in LEVELS:
        ax.plot(thresholds, curves.loc[level][column], color=LEVEL_COLORS[level],
                linewidth=1.4, zorder=3, **kwargs)


def legend(ax, loc):
    ax.legend(
        handles=[Line2D([], [], color=LEVEL_COLORS[k], linewidth=1.4, label=f"{k} consensus")
                 for k in LEVELS],
        frameon=False, loc=loc, handlelength=1.8, borderpad=0, labelspacing=0.25,
    )


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.2), sharex=True)

# (A) What the threshold does to the call sets. The three levels move in
# opposite directions: a stricter threshold cuts edges, so components split and
# 1/3 grows, while the same split strands callers apart and 2/3 and 3/3 shrink.
ax = axes[0, 0]
series(ax, "n_query")
ax.set_yscale("log")
dress(ax, "Consensus calls $\\geq$ 1 kb", "A", bottom=False)
legend(ax, "lower left")

# (B) Precision. Splitting a component tightens its fit to a benchmark interval,
# so every level gains precision as the threshold rises -- 1/3 despite gaining
# calls, 2/3 and 3/3 while losing them.
ax = axes[0, 1]
series(ax, "precision")
ax.set_ylim(0, 1)
dress(ax, "Precision", "B", bottom=False)

# (C) Recall, whose denominator is fixed at the 24,820 benchmark intervals in the
# detectable domain, so the curve is the number of intervals found and nothing
# else. This is the side that pays for the precision in panel B.
ax = axes[1, 0]
series(ax, "recall")
ax.set_ylim(0, 0.35)
dress(ax, "Recall", "C", bottom=True)

# (D) F1, which is where the two directions cancel: flat across the lower half of
# the range for 2/3 and 3/3, rising throughout for 1/3.
ax = axes[1, 1]
series(ax, "f1")
ax.set_ylim(0, 0.4)
dress(ax, "F1", "D", bottom=True)

fig.subplots_adjust(left=0.082, right=0.978, top=0.955, bottom=0.095, wspace=0.21, hspace=0.30)

DEST.mkdir(parents=True, exist_ok=True)
for suffix, dpi in ((".png", 600), (".pdf", None)):
    fig.savefig(DEST / f"consensus_overlap{suffix}", dpi=dpi)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
pd.set_option("display.width", 200)
for level in LEVELS:
    block = curves.loc[level]
    print(f"\n=== {level} ===")
    print(block[["n_query", "median_size", "median_inflation", "p95_inflation",
                 "precision", "recall", "f1", "n_truth_found"]].round(4).to_string())
    print(f"  F1 max {block['f1'].max():.4f} at {block['f1'].idxmax():.2f}"
          f" | precision max {block['precision'].max():.4f} at {block['precision'].idxmax():.2f}"
          f" | recall max {block['recall'].max():.4f} at {block['recall'].idxmax():.2f}")
print(f"\ntruth intervals: {curves['n_truth'].iloc[0]}")
