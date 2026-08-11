# %% Imports and constants

import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from consensuscnv.callsets import (
    collect_callsets,
    merge_components,
    read_bed_calls,
)

from consensuscnv.classification.pairs import build_candidates
from consensuscnv.classification.classify import classify

BENCHMARK_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark")
QUERY_DIR: Path = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/30x_Coverage")
TEST_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/src/test")

# %% Detectable size domain: call sets and sweep parameters
from consensuscnv.classification.intervals import IntervalSet

# Held fixed while the size floor is swept. The floor is justified primarily by
# caller resolution rather than by this curve, so it should not depend on these;
# still worth checking that panel C bends in the same place if they change.
CONSENSUS_THRESHOLD, MIN_SOURCES, PADDING, CLASSIFY_THRESHOLD = 0.5, 2, 0, 0.5
RECOMMENDED_FLOOR = 1_000
TABLE_FLOORS = (1, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000)

# Colourblind-validated pair. Blue is the query side throughout (precision,
# n_query), orange the truth side (recall, n_truth, ceiling), so colour means
# the same thing in every panel.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
QUERY_SIDE, TRUTH_SIDE = "#2a78d6", "#eb6834"

benchmark_callset = collect_callsets(
    read_bed_calls(bed) for bed in glob.glob(str(BENCHMARK_DIR / "*/*.bed"))
)
query_callset = collect_callsets(
    read_bed_calls(bed) for bed in glob.glob(str(QUERY_DIR / "*/*.bed"))
)
truth_full = IntervalSet.from_merged(merge_components(benchmark_callset, max_padding=PADDING))
query_full = IntervalSet.from_merged(
    merge_components(
        query_callset, min_reciprocal_overlap=CONSENSUS_THRESHOLD, min_sources=MIN_SOURCES
    )
)
print(f"{len(truth_full):,} benchmark intervals; {len(query_full):,} query calls")

# %% Sweep a size floor applied to BOTH call sets before matching
# Distinct from the CCDF figure in test_refactoring.py. `size_metrics` with
# COMPLEMENTARY_CUMULATIVE re-aggregates the labels of ONE classification by
# size -- "of calls at or above s, what fraction were true positives". This
# sweep re-runs candidate building and classification on filtered call sets,
# answering "if the detectable domain were DEFINED as >= s, what would the
# metrics be". They differ because removing small truth intervals can strip a
# query call of its only partner, turning a TP into an FP.


def size_domain_sweep(floors):
    """Metrics with both call sets restricted to intervals >= each floor."""
    out = {k: [] for k in ("precision", "recall", "f1", "n_truth", "n_query", "ceiling")}
    for floor in floors:
        truth = truth_full.filter_by_size(min_size=int(floor))
        query = query_full.filter_by_size(min_size=int(floor))
        summary = classify(
            build_candidates(query, truth),
            min_reciprocal_overlap=CLASSIFY_THRESHOLD,
            validate=False,
        ).summary()
        out["precision"].append(summary.precision)
        out["recall"].append(summary.recall)
        out["f1"].append(summary.f1)
        out["n_truth"].append(summary.n_truth)
        out["n_query"].append(summary.n_query)
        # The most recall attainable: every query call matching a distinct truth
        # interval. Recall cannot exceed this however good the callers are.
        out["ceiling"].append(summary.n_query / summary.n_truth if summary.n_truth else np.nan)
    return {k: np.asarray(v) for k, v in out.items()}


# Floors 0 and 1 are identical (no interval is shorter than 1 bp), so the curve
# starts at 1 and its leftmost point is the unrestricted case. A log x-axis
# cannot show 0.
size_floors = np.unique(np.concatenate(([1], np.logspace(0, 5, 80).astype(np.int64))))
size_curve = size_domain_sweep(size_floors)
ceiling_fraction = size_curve["recall"] / size_curve["ceiling"]

# %% Table at key size thresholds
def size_domain_frame(floors):
    """Sweep results as a DataFrame, one row per size floor.

    A floor of 1 bp is the unrestricted case, since no interval is shorter than
    that. Rates are kept as fractions rather than percentages so the frame stays
    usable for arithmetic and exports cleanly.
    """
    sweep = size_domain_sweep(floors)
    frame = pd.DataFrame(
        sweep, index=pd.Index(np.asarray(floors, dtype=np.int64), name="min_size")
    )
    frame["fraction_of_max_recall"] = frame["recall"] / frame["ceiling"]
    return frame.rename(columns={"ceiling": "max_attainable_recall"})[
        [
            "n_truth",
            "n_query",
            "max_attainable_recall",
            "precision",
            "recall",
            "fraction_of_max_recall",
            "f1",
        ]
    ]


size_table = size_domain_frame(TABLE_FLOORS)
size_table.to_csv(TEST_DIR / "detectable_size_domain.tsv", sep="\t")

# %% Figure: choosing a detectable size domain
def dress_size_axis(ax, title, ylabel, last=False):
    ax.set_facecolor(SURFACE)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("bottom", "left"))
        spine.set_color(MUTED)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.set_xscale("log")
    ax.grid(color=MUTED, alpha=0.16, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(RECOMMENDED_FLOOR, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    if last:
        ax.set_xlabel("Minimum CNV size applied to both call sets (bp)", fontsize=9, color=INK_2)


def label_line_end(ax, x, y, text, color):
    """Direct label at the right end, so identity is never colour-alone."""
    ax.annotate(text, (x[-1], y[-1]), textcoords="offset points", xytext=(7, 0),
                fontsize=8.5, color=color, va="center", annotation_clip=False)


fig, axes = plt.subplots(3, 1, figsize=(9, 11), facecolor=SURFACE, sharex=True)

ax = axes[0]
ax.plot(size_floors, size_curve["ceiling"], color=MUTED, linewidth=1.6, linestyle=(0, (5, 2)),
        label="Maximum attainable recall (n_query / n_truth)", zorder=2)
ax.plot(size_floors, size_curve["precision"], color=QUERY_SIDE, linewidth=2,
        label="Precision", zorder=3)
ax.plot(size_floors, size_curve["recall"], color=TRUTH_SIDE, linewidth=2,
        label="Recall", zorder=3)
ax.set_yscale("log")  # precision and recall differ by ~50x
dress_size_axis(ax, "A.  Recall is pinned to a ceiling set by call set sizes, not by detection",
                "Metric (log scale)")
# all three curves are flat on the left, leaving the middle band clear
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="center left")
label_line_end(ax, size_floors, size_curve["precision"], "Precision", QUERY_SIDE)
label_line_end(ax, size_floors, size_curve["recall"], "Recall", TRUTH_SIDE)
ax.annotate(
    f"proposed floor {RECOMMENDED_FLOOR:,} bp\nlargest floor removing no query calls",
    # the clear band between precision above and the recall ceiling below
    xy=(RECOMMENDED_FLOOR, 0.72), xycoords=("data", "axes fraction"),
    textcoords="offset points", xytext=(10, 0), fontsize=8.5, color=INK_2,
    va="center", ha="left",
)

ax = axes[1]
ax.plot(size_floors, size_curve["n_truth"], color=TRUTH_SIDE, linewidth=2,
        label="Benchmark intervals")
ax.plot(size_floors, size_curve["n_query"], color=QUERY_SIDE, linewidth=2, label="Query calls")
ax.set_yscale("log")
dress_size_axis(ax, "B.  The benchmark collapses long before the query set does",
                "Intervals (log scale)")
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower left")
label_line_end(ax, size_floors, size_curve["n_truth"], "Benchmark", TRUTH_SIDE)
label_line_end(ax, size_floors, size_curve["n_query"], "Query", QUERY_SIDE)

ax = axes[2]
ax.plot(size_floors, ceiling_fraction, color=TRUTH_SIDE, linewidth=2)
ax.set_ylim(0, 1)
dress_size_axis(ax, "C.  Fraction of attainable recall achieved -- flat means the restriction "
                    "removes\n     undetectable variants without altering measured performance",
                "Recall / maximum attainable", last=True)
ax.axhline(ceiling_fraction[0], color=MUTED, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
ax.annotate(f"unrestricted: {ceiling_fraction[0]:.1%}", (size_floors[0], ceiling_fraction[0]),
            textcoords="offset points", xytext=(4, 6), fontsize=8.5, color=INK_2)

fig.suptitle("Choosing a detectable size domain", x=0.055, y=0.985, ha="left",
             fontsize=13.5, color=INK)
fig.subplots_adjust(left=0.10, right=0.86, top=0.93, bottom=0.06, hspace=0.30)
fig.savefig(TEST_DIR / "detectable_size_domain.png", dpi=300, facecolor=SURFACE)
