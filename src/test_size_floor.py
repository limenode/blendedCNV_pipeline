# %% Imports and constants

import glob
import os
from itertools import pairwise
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

from consensuscnv.callsets import (
    collect_callsets,
    merge_components,
    read_bed_calls,
)

from consensuscnv.classification.pairs import build_candidates
from consensuscnv.classification.classify import classify

BENCHMARK_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark")
CONTROL_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/SNP_Array")
QUERY_DIRS = {
    "30x": Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/30x_Coverage"),
    "6x": Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/6x_Coverage"),
    "4x": Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/4x_Coverage"),
    "2x": Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/2x_Coverage"),
}
RESULTS_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/results")
CALLERS = ("cnvpytor", "delly", "gatk")
BENCHMARKS = ("1000G", "HGSVC3", "ont_vienna")

TABLE_FLOORS = [1, 100, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]

os.makedirs(str(RESULTS_DIR / "size_floor"), exist_ok=True)


def bed_paths(root: Path, subdirs: tuple[str, ...]) -> list[str]:
    """Every per-sample BED under the named subdirectories of `root`."""
    return [bed for sub in subdirs for bed in sorted(glob.glob(str(root / sub / "*.bed")))]


# %% Detectable size domain: call sets and sweep parameters
from consensuscnv.classification.intervals import IntervalSet

# Held fixed while the size floor is swept.
CONSENSUS_THRESHOLD = 0.5   # query side: reciprocal overlap, no padding
BENCHMARK_PADDING = 0       # truth side: padding only, bridges touching intervals
CLASSIFY_THRESHOLD = 0.5

CONSENSUS_LEVELS = (1, 2, 3)
# Where F1 peaks across the six query sets
RECOMMENDED_DOMAIN = (1_000, 5_000)
REFERENCE_FLOOR = 1_000

# Curves are cut where they fall below this to reduce noise.
MIN_QUERY_CALLS = 100

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
# Three members of the Okabe-Ito qualitative palette (Wong 2011, Nat Methods)
# -- blue, vermillion, and bluish  green are its most separable trio under both
# deuteranopia and protanopia, and it is the palette most journals' figure
# guidance points at.
# Consensus level is ordinal, so it takes a sequential ramp instead; ColorBrewer
# Purples sits clear of all three caller hues and stays dark enough to read on SURFACE
# at the light end. The benchmark is the reference rather than a query set, so it is ink.
LEVEL_COLORS = dict(zip(CONSENSUS_LEVELS, ("#9e9ac8", "#6a51a3", "#3f007d")))
LEVEL_LABELS = {level: f"{level}/3" for level in CONSENSUS_LEVELS}
CALLER_COLORS = {"cnvpytor": "#0072B2", "delly": "#D55E00", "gatk": "#009E73"}
TRUTH_SIDE = INK

os.makedirs(RESULTS_DIR, exist_ok=True)

benchmark_callset = collect_callsets(
    read_bed_calls(bed) for bed in bed_paths(BENCHMARK_DIR, BENCHMARKS)
)
truth_merged = IntervalSet.from_merged(
    merge_components(benchmark_callset, max_padding=BENCHMARK_PADDING)
)

query_callset_dict = {
    name: collect_callsets(read_bed_calls(bed) for bed in bed_paths(query_dir, CALLERS))
    for name, query_dir in QUERY_DIRS.items()
}

# Merged once, at the default min_sources=1. `merge_components` builds components
# from the edge selection alone and only afterwards drops those below min_sources,
# so the components themselves do not depend on it: selecting rows with
# n_sources >= k off the unfiltered result is exactly the set min_sources=k would
# have returned.
query_merged_dict = {
    name: IntervalSet.from_merged(
        merge_components(callset, min_reciprocal_overlap=CONSENSUS_THRESHOLD)
    )
    for name, callset in query_callset_dict.items()
}
query_merged = query_merged_dict["30x"]  # the call set to sweep against the benchmark


def consensus_view(intervals: IntervalSet, min_sources: int) -> IntervalSet:
    """Merged calls supported by at least `min_sources` distinct callers."""
    return intervals.select(intervals.n_sources >= min_sources)


# Each caller is read from its own directory and passed through raw, with no
# merging at all. Selecting a caller's bit out of `query_merged` instead would
# return consensus components that caller took part in, whose extents were set
# partly by the other two callers -- a different quantity.
caller_views = {
    caller: IntervalSet.from_callset(
        collect_callsets(read_bed_calls(bed) for bed in bed_paths(QUERY_DIRS["30x"], (caller,)))
    )
    for caller in CALLERS
}
query_sets = {
    **caller_views,
    **{LEVEL_LABELS[level]: consensus_view(query_merged, level) for level in CONSENSUS_LEVELS},
}
QUERY_COLORS = {**CALLER_COLORS, **{LEVEL_LABELS[k]: LEVEL_COLORS[k] for k in CONSENSUS_LEVELS}}
# Consensus sets carry the argument, so they are drawn heavier than the callers.
QUERY_WIDTHS = {**{c: 1.5 for c in CALLERS}, **{LEVEL_LABELS[k]: 2.2 for k in CONSENSUS_LEVELS}}

print(f"{len(truth_merged):,} benchmark intervals")
for name, view in query_sets.items():
    print(f"  {name}: {len(view):,} query calls")

# %% The one-merge shortcut, checked against three real min_sources= merges
for level in CONSENSUS_LEVELS:
    direct = IntervalSet.from_merged(
        merge_components(
            query_callset_dict["30x"],
            min_reciprocal_overlap=CONSENSUS_THRESHOLD,
            min_sources=level,
        )
    )
    view = query_sets[LEVEL_LABELS[level]]
    assert len(direct) == len(view), (level, len(direct), len(view))
    assert np.array_equal(direct.starts, view.starts)
    assert np.array_equal(direct.ends, view.ends)
    assert np.array_equal(direct.row_index, view.row_index)
print("min_sources shortcut verified against direct merges")


# %% Sweep a size floor applied to BOTH call sets before matching
def size_domain_sweep(floors, query: IntervalSet, truth: IntervalSet = truth_merged):
    """Metrics with both call sets restricted to intervals >= each floor."""
    out = {k: [] for k in ("precision", "recall", "f1", "n_truth", "n_query", "ceiling")}
    for floor in floors:
        summary = classify(
            build_candidates(
                query.filter_by_size(min_size=int(floor)),
                truth.filter_by_size(min_size=int(floor)),
            ),
            min_reciprocal_overlap=CLASSIFY_THRESHOLD,
            validate=False,
        ).summary()
        out["precision"].append(summary.precision)
        out["recall"].append(summary.recall)
        out["f1"].append(summary.f1)
        out["n_truth"].append(summary.n_truth)
        out["n_query"].append(summary.n_query)
        # The most recall attainable: every query call matching a distinct truth
        # interval. Recall cannot exceed this however good the callers are, and
        # it cannot exceed 1 either. The cap earns its place once the larger
        # query sets are on the plot: above a 10 kb floor they hold more calls
        # than the benchmark holds intervals, and past 1 the ratio is no longer
        # a ceiling.
        ceiling = summary.n_query / summary.n_truth if summary.n_truth else np.nan
        out["ceiling"].append(min(ceiling, 1.0))
    return {k: np.asarray(v) for k, v in out.items()}


# Floors 0 and 1 are identical (no interval is shorter than 1 bp), so the curve
# starts at 1 and its leftmost point is the unrestricted case. A log x-axis
# cannot show 0.
size_floors = np.unique(np.concatenate(([1], np.logspace(0, 5, 80).astype(np.int64))))
size_curves = {name: size_domain_sweep(size_floors, view) for name, view in query_sets.items()}

# n_truth is a function of the benchmark and the floor alone, so it has to come
# out identical for every query set. If it ever does not, the floor is being
# applied asymmetrically and the ceiling means nothing.
reference_n_truth = size_curves[next(iter(size_curves))]["n_truth"]
for curve in size_curves.values():
    assert np.array_equal(curve["n_truth"], reference_n_truth)

# recall / ceiling is (n_truth_found / n_truth) / (n_query / n_truth), which is
# n_truth_found / n_query; precision is n_true_positive / n_query. The two agree
# exactly wherever the matching is 1:1, so the fraction-of-attainable-recall
# panel was redrawing the precision panel. Report the largest gap so the claim
# on panel C is quantified rather than asserted: it opens up only where matching
# goes many-to-many, or where the ceiling hit its cap at 1.
ceiling_fractions = {name: c["recall"] / c["ceiling"] for name, c in size_curves.items()}
for name, fraction in ceiling_fractions.items():
    gap = np.abs(size_curves[name]["precision"] - fraction)
    uncapped = size_curves[name]["ceiling"] < 1.0
    print(f"  {name:>8}: max |precision - recall/ceiling| = {np.nanmax(gap[uncapped]):.5f} "
          f"below the cap, {np.nanmax(gap):.4f} overall")


# %% Table at key size thresholds
def size_domain_frame(floors: list[int]) -> pd.DataFrame:
    """Sweep results as a DataFrame, one row per (call set, size floor).

    A floor of 1 bp is the unrestricted case, since no interval is shorter than
    that. Rates are kept as fractions rather than percentages so the frame stays
    usable for arithmetic and exports cleanly.
    """
    frames = []
    for name, view in query_sets.items():
        frame = pd.DataFrame(
            size_domain_sweep(floors, view),
            index=pd.Index(np.asarray(floors, dtype=np.int64), name="min_size"),
        )
        frame.insert(0, "call_set", name)
        frames.append(frame)
    table = pd.concat(frames).set_index("call_set", append=True).swaplevel()
    table["fraction_of_max_recall"] = table["recall"] / table["ceiling"]
    return table.rename(columns={"ceiling": "max_attainable_recall"})[
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
size_table.to_csv(RESULTS_DIR / "size_floor" / "detectable_size_domain.tsv", sep="\t")
print(size_table.to_string(float_format=lambda v: f"{v:.4f}"))

# %% Where the domain comes from, and where the benchmark stops being the larger set
print("\nF1 peak per call set (this is what RECOMMENDED_DOMAIN is set from):")
for name, curve in size_curves.items():
    peak = int(np.nanargmax(curve["f1"]))
    print(f"  {name:>8}: F1 = {curve['f1'][peak]:.3f} at a {size_floors[peak]:,} bp floor")

# The benchmark starts an order of magnitude larger than any query set and ends
# below the largest of them, so there is a crossing. Reporting it keeps panel A's
# title honest -- the benchmark does not simply "collapse first".
for name, curve in size_curves.items():
    below = np.flatnonzero(reference_n_truth < curve["n_query"])
    if below.size:
        print(f"  benchmark falls below {name} above a {size_floors[below[0]]:,} bp floor")

# Every duplication in the merged benchmark, for the caption: after the parser
# stopped folding insertions into DUP this set is almost entirely deletions, so
# the figure is a deletion benchmark and should say so.
from consensuscnv.callsets.registry import SVTYPES

n_dup = int((truth_merged.svtype_idx != SVTYPES.get("DEL")).sum())
print(f"\nbenchmark is {100 * (1 - n_dup / len(truth_merged)):.1f}% deletions "
      f"({n_dup:,} duplications of {len(truth_merged):,} intervals)")


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
    ax.axvspan(*RECOMMENDED_DOMAIN, color=MUTED, alpha=0.10, linewidth=0, zorder=0)
    ax.axvline(REFERENCE_FLOOR, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    if last:
        ax.set_xlabel("Minimum CNV size applied to both call sets (bp)", fontsize=9, color=INK_2)


def trimmed(values, curve):
    """`values` with the low-count tail blanked out.

    A curve is only drawn while the query set behind it still holds at least
    MIN_QUERY_CALLS calls. Blanking rather than slicing keeps every array the
    same length as `size_floors`, so one x vector serves all of them.
    """
    return np.where(curve["n_query"] >= MIN_QUERY_CALLS, values, np.nan)


def label_line_ends(ax, x, entries, min_gap=10.5):
    """Right-edge labels, pushed apart so six converging lines stay readable.

    Identity is never carried by colour alone, but at a 100 kb floor several of
    these curves land within a few pixels of each other, so the labels are
    separated in display space and then converted back to point offsets. Curves
    trimmed for low counts end early, so each label is anchored to that curve's
    last finite point rather than to the right edge of the axis.
    """
    ax.figure.canvas.draw()
    anchored = []
    for y, text, color in entries:
        finite = np.flatnonzero(np.isfinite(y))
        if finite.size:
            last = finite[-1]
            anchored.append([ax.transData.transform((x[last], y[last]))[1],
                             x[last], y[last], text, color])
    placed = sorted(anchored, key=lambda row: row[0])
    for previous, current in pairwise(placed):
        current[0] = max(current[0], previous[0] + min_gap)
    for display_y, data_x, data_y, text, color in placed:
        offset = display_y - ax.transData.transform((data_x, data_y))[1]
        annotation = ax.annotate(
            text, (data_x, data_y), textcoords="offset points", xytext=(7, offset),
            fontsize=8.5, color=color, va="center", annotation_clip=False, zorder=6,
        )
        # Trimmed curves end at different x, so a label can land on top of a
        # neighbouring line rather than in clear space at the right margin.
        # A surface-coloured stroke keeps it legible wherever it falls.
        annotation.set_path_effects(
            [path_effects.withStroke(linewidth=2.6, foreground=SURFACE)]
        )


# Colour carries the call set and line style carries the metric, so the two
# encodings never compete for the same channel. Panel order runs from the cause
# (the benchmark thinning out) to the effect (the metrics it moves).
fig, axes = plt.subplots(4, 1, figsize=(9, 15.5), facecolor=SURFACE, sharex=True)
pending_labels = []

ax = axes[0]
ax.plot(size_floors, reference_n_truth, color=TRUTH_SIDE, linewidth=2,
        linestyle=(0, (5, 2)), zorder=2)
for name, curve in size_curves.items():
    ax.plot(size_floors, curve["n_query"], color=QUERY_COLORS[name],
            linewidth=QUERY_WIDTHS[name], zorder=3)
ax.set_yscale("log")
dress_size_axis(ax, "A.  The benchmark thins out fastest, and ends below the largest query sets",
                "Intervals (log scale)")
pending_labels.append((ax, [(reference_n_truth, "Benchmark", TRUTH_SIDE)]
                       + [(c["n_query"], n, QUERY_COLORS[n]) for n, c in size_curves.items()]))

ax = axes[1]
for name, curve in size_curves.items():
    ax.plot(size_floors, curve["recall"], color=QUERY_COLORS[name],
            linewidth=QUERY_WIDTHS[name], zorder=3)
    ax.plot(size_floors, curve["ceiling"], color=QUERY_COLORS[name], linewidth=1.1,
            linestyle=(0, (5, 2)), alpha=0.7, zorder=2)
ax.set_yscale("log")  # recall and its ceiling differ by an order of magnitude
dress_size_axis(ax, "B.  Recall is pinned to a ceiling set by call set sizes, not by detection",
                "Metric (log scale)")
ax.legend(
    handles=[
        Line2D([], [], color=INK_2, linewidth=2, label="Recall"),
        Line2D([], [], color=INK_2, linewidth=1.1, linestyle=(0, (5, 2)),
               label="Maximum attainable recall (n_query / n_truth, capped at 1)"),
    ],
    frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left",
)
pending_labels.append((ax, [(c["recall"], n, QUERY_COLORS[n]) for n, c in size_curves.items()]))

ax = axes[2]
for name, curve in size_curves.items():
    ax.plot(size_floors, trimmed(curve["precision"], curve), color=QUERY_COLORS[name],
            linewidth=QUERY_WIDTHS[name], zorder=3)
dress_size_axis(ax, "C.  Precision, which is also recall as a fraction of its ceiling:\n"
                    "     recall / maximum attainable recall reduces to "
                    "n_truth_found / n_query",
                "Precision")
pending_labels.append(
    (ax, [(trimmed(c["precision"], c), n, QUERY_COLORS[n]) for n, c in size_curves.items()])
)

# The panel the shaded domain is read off. Every set peaks in the same narrow
# band, which is a stronger argument for the floor than any single set's curve:
# six call sets of very different size and precision agree on where the floor
# should sit.
ax = axes[3]
for name, curve in size_curves.items():
    f1 = trimmed(curve["f1"], curve)
    ax.plot(size_floors, f1, color=QUERY_COLORS[name],
            linewidth=QUERY_WIDTHS[name], zorder=3)
    peak = int(np.nanargmax(f1))
    ax.plot(size_floors[peak], f1[peak], marker="o", markersize=4.5,
            color=QUERY_COLORS[name], markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
dress_size_axis(ax, "D.  F1 peaks in the same narrow band for every call set", "F1", last=True)
pending_labels.append(
    (ax, [(trimmed(c["f1"], c), n, QUERY_COLORS[n]) for n, c in size_curves.items()])
)

fig.suptitle("Choosing a detectable size domain", x=0.055, y=0.988, ha="left",
             fontsize=13.5, color=INK)
fig.subplots_adjust(left=0.10, right=0.83, top=0.945, bottom=0.05, hspace=0.26)
# After subplots_adjust, so the data-to-display transform is final.
for ax, entries in pending_labels:
    label_line_ends(ax, size_floors, entries)
fig.savefig(RESULTS_DIR / "size_floor" / "detectable_size_domain.png", dpi=300, facecolor=SURFACE)
# %% Venn diagram: which callers stand behind each 1/3 consensus call
from matplotlib_venn import venn3
from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm

from consensuscnv.callsets.registry import SOURCES

# The 1/3 set is every component.
one_of_three = query_sets[LEVEL_LABELS[1]]

# `source_bits` is the "unique combination of sources" for a component
caller_bit = {caller: 1 << SOURCES.get(caller) for caller in CALLERS}
caller_mask = sum(caller_bit.values())

assert not (one_of_three.source_bits & ~caller_mask).any(), "non-caller source in query set"

mask_counts = np.bincount(one_of_three.source_bits, minlength=caller_mask + 1)

# venn3 keys read left to right over `CALLERS`, but SOURCES ids are assigned in
# first-appearance order, so the mask is rebuilt from the key
VENN_KEYS = ("100", "010", "110", "001", "101", "011", "111")
venn_regions = {
    key: int(mask_counts[sum(caller_bit[c] for c, on in zip(CALLERS, key) if on == "1")])
    for key in VENN_KEYS
}
assert sum(venn_regions.values()) == len(one_of_three)

venn_frame = pd.DataFrame(
    [
        {
            "callers": "|".join(c for c, on in zip(CALLERS, key) if on == "1"),
            "n_callers": key.count("1"),
            "n_calls": count,
            "pct_of_consensus": 100 * count / len(one_of_three),
        }
        for key, count in venn_regions.items()
    ]
).sort_values(["n_callers", "n_calls"], ascending=[True, False])

os.makedirs(str(RESULTS_DIR / "consensus"), exist_ok=True)
venn_frame.to_csv(RESULTS_DIR / "consensus" / "caller_agreement_30x.tsv", sep="\t", index=False)
print(venn_frame.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# Per-caller totals are component counts
for index, caller in enumerate(CALLERS):
    total = sum(v for k, v in venn_regions.items() if k[index] == "1")
    print(f"  {caller:>8}: {total:,} components carry this caller")

# %% Draw it
# Area-proportional layout puts the 3/3 region at 1/5 the area of cnvpytor-only,
# which is legible; if the ratio ever widens, pass
# layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1,) * 7) to fall
# back to equal circles and let the printed numbers carry the magnitudes.
fig, ax = plt.subplots(figsize=(7.2, 6.0), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

diagram = venn3(
    subsets=venn_regions,
    set_labels=("CNVpytor", "Delly", "GATK-gCNV"),
    set_colors=tuple(CALLER_COLORS[c] for c in CALLERS),
    alpha=0.42,
    ax=ax,
    subset_label_formatter=lambda v: f"{int(v):,}",
    layout_algorithm=DefaultLayoutAlgorithm(),
)
for label in diagram.set_labels:
    if label is not None:
        label.set_fontsize(11)
        label.set_color(INK)
for label in diagram.subset_labels:
    if label is not None:
        label.set_fontsize(9.5)
        label.set_color(INK)
        label.set_path_effects([path_effects.withStroke(linewidth=2.4, foreground=SURFACE)])

ax.set_title(
    f"Caller agreement across {len(one_of_three):,} consensus CNV components, 30x, 13 samples",
    loc="center", fontsize=12, color=INK, pad=12,
)
fig.savefig(
    RESULTS_DIR / "consensus" / "caller_agreement_30x.png", dpi=300,
    facecolor=SURFACE, bbox_inches="tight",
)
