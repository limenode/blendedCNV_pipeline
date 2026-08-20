"""Benchmark-padding sweep behind the choice of zero padding on the truth side.

Where used
----------
Results -> "Parameterizing the Comparison" -> benchmark padding:
    Figure (padding)  the four-panel sweep
    every number quoted in that subsection

Padding is the only parameter that changes what a *truth interval is*. It is
applied to both ends before the benchmark call sets are merged, so a padding of
p bridges any two benchmark intervals separated by at most p and the merged
result inherits the union of their spans.

The subsection's argument is that padding does not act here as a tolerance for
breakpoint imprecision, which is what it is usually reached for. Because the
1 kb size floor is applied *after* merging, padding manufactures truth intervals
out of runs of sub-kilobase fragments that were individually below the detectable
domain. An interval is called *manufactured* here when its longest member record
is itself below the floor, so the interval exists in the truth set only because
padding fused it. Those intervals enter the recall denominator at a detection
rate two orders of magnitude below the rest, so recall falls while the number of
benchmark intervals actually found barely moves.

The script writes two figures. The main one is the four-panel sweep; the
supplement carries the truth-side size distribution and the per-call-set
profiles, which are the same mechanism seen on five more query sets.

    pixi run python manuscript/scripts/benchmark_padding.py
"""

import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, SymmetricalLogLocator

from consensuscnv.callsets import collect_callsets, merge_components, read_bed_calls
from consensuscnv.classification.classify import classify, match_topology
from consensuscnv.classification.intervals import IntervalSet
from consensuscnv.classification.pairs import build_candidates

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")
DEST = ROOT / "results" / "parameterization"
TABLES = ROOT / "results" / "manuscript"

CALLERS = ("cnvpytor", "delly", "gatk")
LABELS = {"cnvpytor": "CNVpytor", "delly": "Delly", "gatk": "GATK-gCNV"}
BENCHMARKS = ("1000G", "HGSVC3", "ont_vienna")
CONSENSUS_LEVELS = (1, 2, 3)

SIZE_FLOOR = 1_000          # the detectable size domain, fixed upstream
CONSENSUS_THRESHOLD = 0.5   # query side, held while padding is swept
CLASSIFY_THRESHOLD = 0.5    # the operating threshold
TOPOLOGY_THRESHOLDS = (0.1, 0.3, 0.5)   # 0.3 and 0.5 go to the CSV only
PANEL_C_THRESHOLD = 0.1
PADDING_CAP = 1_000         # the cap the joint grid is swept to
FOCUS = "2/3"               # the call set drawn in panels C and D

# Okabe-Ito (Wong 2011, Nat Methods). Precision/recall/F1 in panel D and the two
# sides of the matching in panel C are unordered categories, so they take
# categorical hues; the benchmark counts in panels A and B are one quantity seen
# at two levels of restriction and take a grey-to-black ramp instead.
METRIC_COLORS = {"Precision": "#0072B2", "Recall": "#D55E00", "F1": "#009E73"}
QUERY_SIDE, TRUTH_SIDE = "#0072B2", "#D55E00"
ALL_INTERVALS, IN_DOMAIN, FOUND = "#BDBDBD", "#000000", "#D55E00"
# Manufactured intervals are a subset of the >= 1 kb set, so they take the same
# black at a dashed weight rather than a hue of their own.
MANUFACTURED = "#767676"


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
# The benchmark is loaded once. Its overlap and gap edge lists are recorded in
# full at build time and sorted by their own keys, so every padding in the sweep
# is a searchsorted and a slice off this one object rather than a rebuild.
benchmark = collect_callsets(read_bed_calls(bed) for bed in bed_paths(ROOT / "out" / "benchmark", BENCHMARKS))

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
        ).filter_by_size(min_size=SIZE_FLOOR)
        for caller in CALLERS
    },
    **{
        f"{k}/3": consensus.select(consensus.n_sources >= k).filter_by_size(min_size=SIZE_FLOOR)
        for k in CONSENSUS_LEVELS
    },
}
ORDER = ["CNVpytor", "Delly", "GATK-gCNV", "1/3", "2/3", "3/3"]



# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
# Padding 0 is not "off": it bridges exactly-touching intervals, and it is the
# value adopted downstream. `max_padding=None` is off, and is reported as a
# scalar at the end rather than swept, since a log axis has no position for it.
# The decades are added explicitly rather than left to the log grid, so that the
# values quoted in the text and the axis ticks are swept points and not the
# nearest neighbour of one.
DECADES = np.array([0, 10, 100, 1_000, 10_000, 100_000])
paddings = np.unique(np.concatenate((DECADES, np.logspace(0, 5, 70).astype(np.int64))))

# Length of every individual benchmark record, indexed by parent row. A merged
# interval is "manufactured" when the longest record inside it is still below the
# floor: the interval clears the floor only because padding fused the run.
member_lengths = benchmark.ends - benchmark.starts

truth_geometry: list[dict[str, float]] = []
metrics: list[dict[str, float]] = []

for padding in paddings:
    merged = merge_components(benchmark, max_padding=int(padding))
    longest_member = np.zeros(int(merged.labels.max()) + 1, dtype=np.int64)
    np.maximum.at(longest_member, merged.labels, member_lengths)

    intervals = IntervalSet.from_merged(merged)
    in_domain = np.flatnonzero(intervals.lengths >= SIZE_FLOOR)
    truth = intervals.select(in_domain)
    manufactured = longest_member[merged.component_id][in_domain] < SIZE_FLOOR

    lengths = truth.lengths
    quartiles = np.percentile(lengths, [25, 50, 75]) if len(lengths) else [np.nan] * 3
    truth_geometry.append({
        "padding": int(padding),
        "n_truth_all": len(intervals),
        "n_truth": len(truth),
        "n_manufactured": int(manufactured.sum()),
        # The whole merged set and the part of it inside the detectable domain
        # move in opposite directions, so both medians are worth carrying.
        "median_all": float(np.median(intervals.lengths)),
        "q1": float(quartiles[0]), "median": float(quartiles[1]), "q3": float(quartiles[2]),
    })

    for name in ORDER:
        candidates = build_candidates(query_sets[name], truth)
        for threshold in TOPOLOGY_THRESHOLDS:
            classification = classify(candidates, min_reciprocal_overlap=threshold, validate=False)
            summary, topology = classification.summary(), match_topology(classification)
            # Per-row found mask, so the two classes of truth interval can be
            # scored separately. Everything that is not a false negative was
            # matched by at least one query call.
            found = np.ones(len(truth), dtype=bool)
            found[classification.false_negative_rows] = False
            metrics.append({
                "call_set": name, "classify_threshold": threshold, "padding": int(padding),
                "precision": summary.precision, "recall": summary.recall, "f1": summary.f1,
                "n_query": summary.n_query, "n_truth": summary.n_truth,
                "n_true_positive": summary.n_true_positive,
                "n_truth_found": classification.n_truth_found,
                "found_rate_native": float(found[~manufactured].mean()) if (~manufactured).any() else np.nan,
                "found_rate_manufactured": float(found[manufactured].mean()) if manufactured.any() else np.nan,
                "n_pairs": topology.n_pairs,
                "n_query_multi": topology.n_query_multi,
                "n_truth_multi": topology.n_truth_multi,
                "max_query_partners": topology.max_query_partners,
                "max_truth_partners": topology.max_truth_partners,
            })

geometry = pd.DataFrame(truth_geometry).set_index("padding")
curves = pd.DataFrame(metrics).set_index(["call_set", "classify_threshold", "padding"]).sort_index()

# The query side is untouched by padding, so every precision denominator in the
# sweep has to be the one count. If it ever is not, padding is leaking into the
# query merge and the whole comparison below is between moving targets.
for name in ORDER:
    assert curves.loc[name]["n_query"].nunique() == 1, name

TABLES.mkdir(parents=True, exist_ok=True)
geometry.to_csv(TABLES / "benchmark_padding_truth.csv")
curves.to_csv(TABLES / "benchmark_padding_curves.csv")

focus = curves.loc[(FOCUS, CLASSIFY_THRESHOLD)]


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

PAD_TICKS = {0: "0", 1e1: "10 bp", 1e2: "100 bp", 1e3: "1 kb", 1e4: "10 kb", 1e5: "100 kb"}


def dress(ax, ylabel: str, letter: str, bottom: bool) -> None:
    # Padding 0 is a real setting and the one adopted, so it has to appear. A log
    # axis has no position for it; symlog gives the [0, 1] interval a linear
    # segment, compressed by linscale so it does not read as a whole decade.
    ax.set_xscale("symlog", linthresh=1, linscale=0.35)
    ax.set_xlim(0, 1e5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel, labelpad=2)
    # The cap on the joint grid, drawn light and dotted on every panel so the
    # same vertical reference reads across all four without competing with data.
    ax.axvline(PADDING_CAP, color="#B0B0B0", linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
    ax.xaxis.set_major_locator(FixedLocator(list(PAD_TICKS)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: PAD_TICKS.get(v, "")))
    ax.xaxis.set_minor_locator(
        SymmetricalLogLocator(base=10, linthresh=1, subs=list(np.arange(2, 10)))
    )
    ax.tick_params(which="minor", length=2.0)
    ax.tick_params(which="major", length=3.4)
    # Every panel carries its own padding scale. The axes are shared, so the tick
    # labels have to be switched back on for the top row; the axis title stays on
    # the bottom row alone rather than being repeated four times.
    ax.tick_params(axis="x", labelbottom=True)
    if bottom:
        ax.set_xlabel("Benchmark padding, applied to both ends of every interval", labelpad=2)
    ax.text(-0.155, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def legend(ax, handles, loc):
    ax.legend(handles=handles, frameon=False, loc=loc, handlelength=1.8, borderpad=0,
              labelspacing=0.25)


# --------------------------------------------------------------------------- #
# Main figure
# --------------------------------------------------------------------------- #
# 180 mm is the standard full-width figure; the four panels sit in a 2x2 grid so
# the whole sweep is read at one glance rather than scrolled.
fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.2), sharex=True)

# (A) What padding does to the truth set. The four curves move in four different
# ways, and that is the mechanism: padding destroys intervals overall, creates
# them inside the detectable domain, and almost all of the created ones are
# fused runs of fragments.
ax = axes[0, 0]
ax.plot(paddings, geometry["n_truth_all"], color=ALL_INTERVALS, linewidth=1.2, zorder=2)
ax.plot(paddings, geometry["n_truth"], color=IN_DOMAIN, linewidth=1.4, zorder=3)
ax.plot(paddings, geometry["n_manufactured"], color=MANUFACTURED, linewidth=1.4,
        linestyle=(0, (4, 2)), zorder=3)
ax.plot(paddings, focus["n_truth_found"], color=FOUND, linewidth=1.4, zorder=4)
ax.set_yscale("log")
# Room below the manufactured curve's starting value for the legend, which has
# nowhere else to sit: the four curves between them occupy the whole upper panel.
ax.set_ylim(4e1, 4e5)
dress(ax, "Benchmark intervals", "A", bottom=False)
legend(ax, [
    Line2D([], [], color=ALL_INTERVALS, linewidth=1.2, label="All merged"),
    Line2D([], [], color=IN_DOMAIN, linewidth=1.4, label="$\\geq$ 1 kb (recall denominator)"),
    Line2D([], [], color=MANUFACTURED, linewidth=1.4, linestyle=(0, (4, 2)),
           label="of which manufactured by padding"),
    Line2D([], [], color=FOUND, linewidth=1.4, label="Found by the 2-of-3 consensus"),
], "lower right")

# (B) And whether the created intervals are reachable. They are not: their
# detection rate sits two orders of magnitude below that of the intervals the
# benchmark already contained, which is why panel D's recall falls.
ax = axes[0, 1]
ax.plot(paddings, 100 * focus["found_rate_native"], color=IN_DOMAIN, linewidth=1.4, zorder=3)
ax.plot(paddings, 100 * focus["found_rate_manufactured"], color=MANUFACTURED, linewidth=1.4,
        linestyle=(0, (4, 2)), zorder=3)
ax.set_yscale("log")
ax.set_ylim(5e-3, 1e2)
dress(ax, "Benchmark intervals found (%)", "B", bottom=False)
legend(ax, [
    Line2D([], [], color=IN_DOMAIN, linewidth=1.4, label="Native intervals"),
    Line2D([], [], color=MANUFACTURED, linewidth=1.4, linestyle=(0, (4, 2)),
           label="Manufactured intervals"),
], "center left")

# (C) Whether padding changes the shape of the matching. At the operating
# threshold both counts are identically zero at every padding, so those two lines
# lie on the axis -- that flatness is the result, not a rendering accident.
ax = axes[1, 0]
# Drawn at a permissive 0.1 only. At the 0.5 threshold used throughout, a query
# call cannot reach half of two benchmark intervals that do not themselves
# overlap, so both counts are zero by construction rather than by measurement and
# plotting them would put two flat lines on the axis for no information.
row = curves.loc[(FOCUS, PANEL_C_THRESHOLD)]
ax.plot(paddings, row["n_query_multi"], color=QUERY_SIDE, linewidth=1.2, zorder=3)
ax.plot(paddings, row["n_truth_multi"], color=TRUTH_SIDE, linewidth=1.2, zorder=3)
# Headroom for the legend, which would otherwise sit on the query-side curve.
ax.set_ylim(-1, 62)
dress(ax, "Rows with more than one partner", "C", bottom=True)
legend(ax, [
    Line2D([], [], color=QUERY_SIDE, linewidth=1.2, label="Query calls spanning $>$1 benchmark interval"),
    Line2D([], [], color=TRUTH_SIDE, linewidth=1.2, label="Benchmark intervals split across $>$1 query call"),
], "upper right")

# (D) The consequence for the metrics. Recall falls across the whole sweep even
# though panel A shows the number found is flat, because only its denominator is
# moving.
ax = axes[1, 1]
for label, column in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")):
    ax.plot(paddings, focus[column], color=METRIC_COLORS[label], linewidth=1.4, zorder=3)
ax.set_ylim(0, 1)
dress(ax, "Classification metric", "D", bottom=True)
legend(ax, [Line2D([], [], color=c, linewidth=1.4, label=n) for n, c in METRIC_COLORS.items()],
       "upper right")

fig.subplots_adjust(left=0.082, right=0.978, top=0.955, bottom=0.095, wspace=0.21, hspace=0.30)

DEST.mkdir(parents=True, exist_ok=True)
for suffix, dpi in ((".png", 600), (".pdf", None)):
    fig.savefig(DEST / f"benchmark_padding{suffix}", dpi=dpi)


# --------------------------------------------------------------------------- #
# Supplementary figure: the truth-side size distribution, and the same three
# metrics on all six 30x query call sets rather than on the one carried
# downstream.
# --------------------------------------------------------------------------- #
# Okabe-Ito for the callers, a sequential Purples ramp for the ordinal consensus
# levels -- the palette of the size-domain figure, so the six call sets keep one
# identity across the paper.
SET_COLORS = {"CNVpytor": "#0072B2", "Delly": "#D55E00", "GATK-gCNV": "#009E73",
              "1/3": "#9E9AC8", "2/3": "#6A51A3", "3/3": "#3F007D"}
SET_WIDTHS = {**{LABELS[c]: 1.0 for c in CALLERS}, **{f"{k}/3": 1.5 for k in CONSENSUS_LEVELS}}

sfig, saxes = plt.subplots(2, 2, figsize=(7.09, 5.2), sharex=True)

ax = saxes[0, 0]
ax.fill_between(paddings, geometry["q1"], geometry["q3"], color=IN_DOMAIN, alpha=0.12,
                linewidth=0, zorder=2)
ax.plot(paddings, geometry["median"], color=IN_DOMAIN, linewidth=1.4, zorder=3)
ax.plot(paddings, geometry["median_all"], color=ALL_INTERVALS, linewidth=1.2, zorder=2)
ax.set_yscale("log")
dress(ax, "Benchmark interval size (bp)", "A", bottom=False)
legend(ax, [
    Line2D([], [], color=IN_DOMAIN, linewidth=1.4, label="Median, $\\geq$ 1 kb (IQR shaded)"),
    Line2D([], [], color=ALL_INTERVALS, linewidth=1.2, label="Median, all merged"),
], "upper left")

for ax, letter, column, ylabel, ylim in (
    (saxes[0, 1], "B", "precision", "Precision", (0, 1)),
    (saxes[1, 0], "C", "recall", "Recall", (0, None)),
    (saxes[1, 1], "D", "f1", "F1", (0, None)),
):
    for name in ORDER:
        ax.plot(paddings, curves.loc[(name, CLASSIFY_THRESHOLD)][column], color=SET_COLORS[name],
                linewidth=SET_WIDTHS[name], zorder=3)
    ax.set_ylim(*ylim)
    dress(ax, ylabel, letter, bottom=letter in ("C", "D"))

sfig.legend(
    handles=[Line2D([], [], color=SET_COLORS[n], linewidth=SET_WIDTHS[n], label=n) for n in ORDER],
    loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.005),
    handlelength=1.8, columnspacing=1.4, handletextpad=0.5,
)
sfig.subplots_adjust(left=0.082, right=0.978, top=0.955, bottom=0.135, wspace=0.21, hspace=0.30)
for suffix, dpi in ((".png", 600), (".pdf", None)):
    sfig.savefig(DEST / f"benchmark_padding_supplement{suffix}", dpi=dpi)


# --------------------------------------------------------------------------- #
# The numbers quoted alongside the figure
# --------------------------------------------------------------------------- #
def at(padding: int) -> pd.Series:
    return geometry.loc[padding]


unpadded = IntervalSet.from_merged(merge_components(benchmark, max_padding=None))
print(f"padding off entirely: {len(unpadded):,} merged intervals, "
      f"{len(unpadded.filter_by_size(min_size=SIZE_FLOOR)):,} at >= 1 kb")
print(f"padding 0 (touching bridged): {at(0)['n_truth_all']:,.0f} merged, {at(0)['n_truth']:,.0f} at >= 1 kb")

print("\nall merged / >= 1 kb / manufactured / median of the >= 1 kb set:")
for padding in DECADES:
    row = at(int(padding))
    print(f"  {padding:>7,} bp: {row['n_truth_all']:>8,.0f} {row['n_truth']:>8,.0f} "
          f"{row['n_manufactured']:>7,.0f} ({100 * row['n_manufactured'] / row['n_truth']:>4.1f}%) "
          f"{row['median']:>8,.0f} bp [{row['q1']:,.0f}-{row['q3']:,.0f}]")

trough = int(geometry["median"].idxmin())
print(f"\nmedian of the >= 1 kb set is lowest at {trough:,} bp padding "
      f"({at(trough)['median']:,.0f} bp, from {at(0)['median']:,.0f} bp unpadded)")

print(f"\n{FOCUS} consensus at 30x, classification threshold {CLASSIFY_THRESHOLD}:")
for padding in DECADES:
    row = focus.loc[int(padding)]
    print(f"  {padding:>7,} bp: P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} "
          f"found={row['n_truth_found']:,.0f} of {row['n_truth']:,.0f} | "
          f"found rate native {100 * row['found_rate_native']:.2f}% "
          f"vs manufactured {100 * row['found_rate_manufactured']:.2f}%")

print("\nrecall lost to padding, decomposed (numerator vs denominator):")
base, capped = focus.loc[0], focus.loc[PADDING_CAP]
print(f"  found {base['n_truth_found']:,.0f} -> {capped['n_truth_found']:,.0f} "
      f"({100 * (capped['n_truth_found'] / base['n_truth_found'] - 1):+.1f}%), "
      f"denominator {base['n_truth']:,.0f} -> {capped['n_truth']:,.0f} "
      f"({100 * (capped['n_truth'] / base['n_truth'] - 1):+.1f}%), "
      f"recall {base['recall']:.3f} -> {capped['recall']:.3f}")

print("\nmatch topology across the sweep (max over paddings):")
for threshold in TOPOLOGY_THRESHOLDS:
    row = curves.loc[(FOCUS, threshold)]
    print(f"  threshold {threshold}: query rows with >1 partner <= {row['n_query_multi'].max():,.0f}, "
          f"truth rows with >1 partner <= {row['n_truth_multi'].max():,.0f}, "
          f"of {row['n_pairs'].min():,.0f}-{row['n_pairs'].max():,.0f} pairs")

print("\nprecision and F1 across all six 30x call sets, padding 0 -> 1 kb -> 100 kb:")
for name in ORDER:
    row = curves.loc[(name, CLASSIFY_THRESHOLD)]
    print(f"  {name:>9}: P {row.loc[0, 'precision']:.3f} -> {row.loc[PADDING_CAP, 'precision']:.3f} "
          f"-> {row.loc[100_000, 'precision']:.3f} | "
          f"R {row.loc[0, 'recall']:.3f} -> {row.loc[PADDING_CAP, 'recall']:.3f} "
          f"-> {row.loc[100_000, 'recall']:.3f}")

print(f"\nwrote {DEST / 'benchmark_padding.png'} and {DEST / 'benchmark_padding_supplement.png'}")
