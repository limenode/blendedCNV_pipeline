"""Size-floor sweep behind the choice of a detectable size domain.

Where used
----------
Results -> "Choosing a Detectable Size Domain":
    Figure 4  the four-panel sweep
    every number quoted in that section (crossings, ceilings, F1 peaks)

A size floor is applied symmetrically to the 30x query call sets and to the
merged benchmark, and the comparison is rerun at each floor. The floor has to be
symmetric: dropping small benchmark intervals alone would inflate precision by
removing exactly the intervals the callers cannot reach.

The point of the sweep is that the usable domain is bounded from below by the
caller bin size and from above by the benchmark running out of intervals, and
that both bounds are visible in the data rather than asserted.

This is the publication rendering. `src/test_size_floor.py` holds the working
version of the same sweep, with the wider diagnostics used to develop it.

    pixi run python manuscript/scripts/detectable_size_domain.py
"""

import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

from consensuscnv.callsets import collect_callsets, merge_components, read_bed_calls
from consensuscnv.classification.classify import classify
from consensuscnv.classification.intervals import IntervalSet
from consensuscnv.classification.pairs import build_candidates

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")
DEST = ROOT / "results" / "size_floor"
TABLES = ROOT / "results" / "manuscript"

CALLERS = ("cnvpytor", "delly", "gatk")
LABELS = {"cnvpytor": "CNVpytor", "delly": "Delly", "gatk": "GATK-gCNV"}
BENCHMARKS = ("1000G", "HGSVC3", "ont_vienna")
CONSENSUS_LEVELS = (1, 2, 3)

CONSENSUS_THRESHOLD = 0.5   # query side: reciprocal overlap, no padding
BENCHMARK_PADDING = 0       # truth side: padding only, bridges touching intervals
CLASSIFY_THRESHOLD = 0.5

RECOMMENDED_DOMAIN = (1_000, 5_000)   # spans every F1 maximum
REFERENCE_FLOOR = 1_000               # the floor adopted downstream
MIN_QUERY_CALLS = 100                 # curves are cut below this

# Okabe-Ito (Wong 2011, Nat Methods) for the callers -- its most separable trio
# under both deuteranopia and protanopia. Consensus level is ordinal, so it takes
# a sequential ramp (ColorBrewer Purples) that sits clear of all three caller
# hues. The benchmark is the reference rather than a query set, so it is black.
COLORS = {
    "CNVpytor": "#0072B2", "Delly": "#D55E00", "GATK-gCNV": "#009E73",
    "1/3": "#9E9AC8", "2/3": "#6A51A3", "3/3": "#3F007D",
}
BENCHMARK_COLOR = "#000000"
# Consensus sets carry the argument of the section, so they are drawn heavier.
WIDTHS = {**{LABELS[c]: 1.0 for c in CALLERS}, **{f"{k}/3": 1.5 for k in CONSENSUS_LEVELS}}
ORDER = ["CNVpytor", "Delly", "GATK-gCNV", "1/3", "2/3", "3/3"]


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
)

coverage_dir = ROOT / "out" / "30x_Coverage"
# Merged once at the default min_sources=1: components are built from the edge
# selection alone and only afterwards dropped, so selecting n_sources >= k off
# the unfiltered result is exactly the set min_sources=k would have returned.
consensus = IntervalSet.from_merged(
    merge_components(
        collect_callsets(read_bed_calls(bed) for bed in bed_paths(coverage_dir, CALLERS)),
        min_reciprocal_overlap=CONSENSUS_THRESHOLD,
    )
)
# Each caller is read from its own directory and passed through raw. Selecting a
# caller's bit out of `consensus` instead would return components that caller
# took part in, whose extents were set partly by the other two -- a different
# quantity.
query_sets = {
    **{
        LABELS[caller]: IntervalSet.from_callset(
            collect_callsets(read_bed_calls(bed) for bed in bed_paths(coverage_dir, (caller,)))
        )
        for caller in CALLERS
    },
    **{f"{k}/3": consensus.select(consensus.n_sources >= k) for k in CONSENSUS_LEVELS},
}


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def sweep(floors: np.ndarray, query: IntervalSet) -> dict[str, np.ndarray]:
    """Metrics with both call sets restricted to intervals >= each floor."""
    out: dict[str, list[float]] = {k: [] for k in ("precision", "recall", "f1", "n_truth", "n_query", "ceiling")}
    for floor in floors:
        summary = classify(
            build_candidates(query.filter_by_size(min_size=int(floor)), truth.filter_by_size(min_size=int(floor))),
            min_reciprocal_overlap=CLASSIFY_THRESHOLD,
            validate=False,
        ).summary()
        for key in ("precision", "recall", "f1", "n_truth", "n_query"):
            out[key].append(getattr(summary, key))
        # The most recall attainable: every query call matching a distinct truth
        # interval. Recall cannot exceed this however good the callers are, and
        # it cannot exceed 1 either -- above a 10 kb floor the larger query sets
        # hold more calls than the benchmark holds intervals.
        out["ceiling"].append(min(summary.n_query / summary.n_truth, 1.0) if summary.n_truth else np.nan)
    return {k: np.asarray(v) for k, v in out.items()}


# Floors 0 and 1 are identical (no interval is shorter than 1 bp), so the sweep
# starts at 1 and its leftmost point is the unrestricted case; a log axis cannot
# show 0.
floors = np.unique(np.concatenate(([1], np.logspace(0, 5, 80).astype(np.int64))))
curves = {name: sweep(floors, view) for name, view in query_sets.items()}

# n_truth is a function of the benchmark and the floor alone, so it has to come
# out identical for every query set. If it ever does not, the floor is being
# applied asymmetrically and the ceiling means nothing.
n_truth = curves[ORDER[0]]["n_truth"]
for curve in curves.values():
    assert np.array_equal(curve["n_truth"], n_truth)

TABLES.mkdir(parents=True, exist_ok=True)
pd.concat(
    [pd.DataFrame(curve).assign(call_set=name, min_size=floors) for name, curve in curves.items()]
).set_index(["call_set", "min_size"]).to_csv(TABLES / "detectable_size_domain_curves.csv")


def trimmed(values: np.ndarray, curve: dict[str, np.ndarray]) -> np.ndarray:
    """`values` with the low-count tail blanked out.

    Precision estimated from a few dozen calls is not comparable with precision
    estimated from thousands. Blanking rather than slicing keeps every array the
    same length as `floors`, so one x vector serves all of them.
    """
    return np.where(curve["n_query"] >= MIN_QUERY_CALLS, values, np.nan)


# --------------------------------------------------------------------------- #
# Figure
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
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 6.5,
    "lines.solid_capstyle": "round",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,   # embed TrueType rather than Type 3, per journal guidance
    "ps.fonttype": 42,
})

SIZE_TICKS = [(1, "1 bp"), (1e1, "10 bp"), (1e2, "100 bp"), (1e3, "1 kb"), (1e4, "10 kb"), (1e5, "100 kb")]

# 180 mm is the standard full-width figure; the four panels sit in a 2x2 grid so
# the whole sweep is read at one glance rather than scrolled.
fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.2), sharex=True)


def dress(ax, ylabel: str, letter: str, bottom: bool) -> None:
    ax.set_xscale("log")
    ax.set_xlim(1, 1e5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel, labelpad=2)
    # The adopted floor and the band spanning every F1 maximum, on every panel so
    # the same vertical reference reads across all four. Both are kept light and
    # the floor is dotted rather than dashed: the benchmark in panel A and the
    # recall ceilings in panel B are dashed, and a third dashed line would read
    # as a fourth data series rather than as an annotation.
    ax.axvspan(*RECOMMENDED_DOMAIN, color="#000000", alpha=0.05, linewidth=0, zorder=0)
    ax.axvline(REFERENCE_FLOOR, color="#B0B0B0", linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: dict(SIZE_TICKS).get(v, "")))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1), numticks=12))
    ax.tick_params(which="minor", length=2.0)
    ax.tick_params(which="major", length=3.4)
    # Every panel carries its own size scale. The axes are shared, so the tick
    # labels have to be switched back on for the top row; the axis title stays on
    # the bottom row alone rather than being repeated four times.
    ax.tick_params(axis="x", labelbottom=True)
    if bottom:
        ax.set_xlabel("Minimum CNV size applied to both call sets", labelpad=2)
    ax.text(-0.155, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


ax = axes[0, 0]
ax.plot(floors, n_truth, color=BENCHMARK_COLOR, linewidth=1.1, linestyle=(0, (4, 2)), zorder=4)
for name in ORDER:
    ax.plot(floors, curves[name]["n_query"], color=COLORS[name], linewidth=WIDTHS[name], zorder=3)
ax.set_yscale("log")
ax.set_ylim(30, 4e5)
dress(ax, "Intervals surviving the floor", "A", bottom=False)

ax = axes[0, 1]
for name in ORDER:
    ax.plot(floors, curves[name]["ceiling"], color=COLORS[name], linewidth=0.7,
            linestyle=(0, (3, 2)), alpha=0.85, zorder=2)
    ax.plot(floors, curves[name]["recall"], color=COLORS[name], linewidth=WIDTHS[name], zorder=3)
ax.set_yscale("log")   # recall and its ceiling differ by an order of magnitude
dress(ax, "Recall", "B", bottom=False)
ax.legend(
    handles=[
        Line2D([], [], color="#4D4D4D", linewidth=1.2, label="Recall"),
        Line2D([], [], color="#4D4D4D", linewidth=0.7, linestyle=(0, (3, 2)),
               label="Maximum attainable recall"),
    ],
    frameon=False, loc="upper left", handlelength=2.2, borderpad=0, labelspacing=0.25,
)

ax = axes[1, 0]
for name in ORDER:
    ax.plot(floors, trimmed(curves[name]["precision"], curves[name]), color=COLORS[name],
            linewidth=WIDTHS[name], zorder=3)
ax.set_ylim(0, 1)
dress(ax, "Precision", "C", bottom=True)

ax = axes[1, 1]
for name in ORDER:
    f1 = trimmed(curves[name]["f1"], curves[name])
    ax.plot(floors, f1, color=COLORS[name], linewidth=WIDTHS[name], zorder=3)
    peak = int(np.nanargmax(f1))
    ax.plot(floors[peak], f1[peak], marker="o", markersize=3, color=COLORS[name],
            markeredgecolor="white", markeredgewidth=0.5, zorder=4)
ax.set_ylim(0, None)
dress(ax, "F1", "D", bottom=True)

fig.legend(
    handles=[Line2D([], [], color=COLORS[n], linewidth=WIDTHS[n], label=n) for n in ORDER]
    + [Line2D([], [], color=BENCHMARK_COLOR, linewidth=1.1, linestyle=(0, (4, 2)), label="Benchmark")],
    loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.005),
    handlelength=1.8, columnspacing=1.4, handletextpad=0.5,
)
fig.subplots_adjust(left=0.082, right=0.978, top=0.955, bottom=0.135, wspace=0.19, hspace=0.30)

DEST.mkdir(parents=True, exist_ok=True)
for suffix, dpi in ((".png", 600), (".pdf", None)):
    fig.savefig(DEST / f"detectable_size_domain_pub{suffix}", dpi=dpi)


# --------------------------------------------------------------------------- #
# The numbers quoted alongside the figure
# --------------------------------------------------------------------------- #
print(f"{len(truth):,} benchmark intervals; " + ", ".join(f"{n} {len(v):,}" for n, v in query_sets.items()))

print("\nfloor above which the benchmark holds fewer intervals than the query set:")
for name in ORDER:
    below = np.flatnonzero(n_truth < curves[name]["n_query"])
    print(f"  {name:>9}: {f'{floors[below[0]]:,} bp' if below.size else 'never'}")

print("\nmaximum attainable recall at an unrestricted floor, and F1 peak:")
for name in ORDER:
    curve, f1 = curves[name], trimmed(curves[name]["f1"], curves[name])
    peak = int(np.nanargmax(f1))
    saturates = np.flatnonzero(curve["ceiling"] >= 1.0)
    print(f"  {name:>9}: ceiling {curve['ceiling'][0]:.3f}, max ceiling {np.nanmax(curve['ceiling']):.4f}"
          f"{f', saturates above {floors[saturates[0]]:,} bp' if saturates.size else ''}"
          f" | F1 {f1[peak]:.3f} at {floors[peak]:,} bp")

print("\nprecision between the unrestricted case and a 100 kb floor:")
for name in ORDER:
    precision = trimmed(curves[name]["precision"], curves[name])
    finite = np.flatnonzero(np.isfinite(precision))
    print(f"  {name:>9}: {precision[finite[0]]:.3f} -> {precision[finite[-1]]:.3f} "
          f"(last drawn at a {floors[finite[-1]]:,} bp floor)")

at_reference = int(np.searchsorted(floors, REFERENCE_FLOOR))
print(f"\ntruth intervals: {n_truth[0]:,} unrestricted -> {n_truth[at_reference]:,} "
      f"at the nearest swept floor ({floors[at_reference]:,} bp); "
      f"{len(truth.filter_by_size(min_size=REFERENCE_FLOOR)):,} at exactly 1 kb")
print(f"wrote {DEST / 'detectable_size_domain_pub.png'}")
