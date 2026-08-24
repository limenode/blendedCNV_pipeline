"""Classification reciprocal-overlap sweep behind the choice of 0.5.

Where used
----------
Results -> "Parameterizing the Comparison" -> classification reciprocal overlap:
    Figure (classification overlap)  the four-panel sweep
    every number quoted in that subsection

Unlike the other three parameters this one changes neither call set. Both sides
are fixed (benchmark at zero padding, 1 kb floor on each side, consensus built
at 0.5) and only the rule crediting a query call against a benchmark interval
moves, so the whole sweep is a filter over one candidate-pair list per call set.

Two things are worth separating in the result. The metrics decline
monotonically, so the threshold has no interior optimum and cannot be selected
by maximising anything. What does change qualitatively is the *topology*: at or
above 0.5 a query call cannot reach half of each of two benchmark intervals
unless those intervals overlap, so the matching is forced one-to-one; below it
the same call is credited against several intervals at once and the query-side
and truth-side true-positive counts come apart.

    pixi run python manuscript/scripts/classification_overlap.py
"""

import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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

SIZE_FLOOR = 1_000            # the detectable size domain, fixed upstream
BENCHMARK_PADDING = 0         # bridges exactly-touching records, nothing wider
CONSENSUS_THRESHOLD = 0.5     # held while the classification threshold is swept
ADOPTED = 0.5                 # the value this sweep is asked to justify
FOCUS = "2/3"                 # the call set drawn in panels A-C

# Okabe-Ito (Wong 2011, Nat Methods) for the unordered categories, a sequential
# Purples ramp for the ordinal consensus levels. Same assignment as the padding
# and size-domain figures.
METRIC_COLORS = {"Precision": "#0072B2", "Recall": "#D55E00", "F1": "#009E73"}
QUERY_SIDE, TRUTH_SIDE = "#0072B2", "#D55E00"
SET_COLORS = {"CNVpytor": "#0072B2", "Delly": "#D55E00", "GATK-gCNV": "#009E73",
              "1/3": "#9E9AC8", "2/3": "#6A51A3", "3/3": "#3F007D"}
SET_WIDTHS = {**{LABELS[c]: 1.0 for c in CALLERS}, **{f"{k}/3": 1.5 for k in CONSENSUS_LEVELS}}
ORDER = ["CNVpytor", "Delly", "GATK-gCNV", "1/3", "2/3", "3/3"]


def bed_paths(root: Path, subdirs: tuple[str, ...]) -> list[str]:
    """Every per-sample BED under the named subdirectories of `root`.

    Pinned to an explicit tuple rather than a wildcard: consensus output is
    destined to land beside the per-caller directories, and a glob would read it
    back in as a fourth source.
    """
    return [bed for sub in subdirs for bed in sorted(glob.glob(str(root / sub / "*.bed")))]


# --------------------------------------------------------------------------- #
# Call sets -- both sides fixed, built once
# --------------------------------------------------------------------------- #
truth = IntervalSet.from_merged(
    merge_components(
        collect_callsets(read_bed_calls(bed) for bed in bed_paths(ROOT / "out" / "benchmark", BENCHMARKS)),
        max_padding=BENCHMARK_PADDING,
    )
).filter_by_size(min_size=SIZE_FLOOR)

coverage_dir = ROOT / "out" / "30x_Coverage"
consensus = IntervalSet.from_merged(
    merge_components(
        collect_callsets(read_bed_calls(bed) for bed in bed_paths(coverage_dir, CALLERS)),
        min_reciprocal_overlap=CONSENSUS_THRESHOLD,
    )
)
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

# Zero is a real setting -- one shared base pair counts as a match -- and is the
# left endpoint of the sweep even though it is excluded from the joint grid.
thresholds = np.round(np.arange(0.0, 1.00, 0.01), 2)


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
# One candidate list per call set. Every pair that could ever be credited is
# recorded at build time, sorted by reciprocal overlap, so each threshold is a
# searchsorted and a slice rather than a rebuild.
rows: list[dict[str, float]] = []

for name in ORDER:
    candidates = build_candidates(query_sets[name], truth)
    for threshold in thresholds:
        classification = classify(candidates, min_reciprocal_overlap=float(threshold), validate=False)
        summary, topology = classification.summary(), match_topology(classification)
        rows.append({
            "call_set": name,
            "classify_threshold": float(threshold),
            "n_query": summary.n_query,
            "n_truth": summary.n_truth,
            "precision": summary.precision,
            "recall": summary.recall,
            "f1": summary.f1,
            # Query rows matched and truth rows found are different numbers
            # whenever the matching is many-to-many; they are the two sides the
            # 0.5 threshold forces together.
            "n_true_positive": summary.n_true_positive,
            "n_truth_found": summary.n_truth_found,
            "n_pairs": topology.n_pairs,
            "n_query_multi": topology.n_query_multi,
            "n_truth_multi": topology.n_truth_multi,
            "max_query_partners": topology.max_query_partners,
            "max_truth_partners": topology.max_truth_partners,
        })

curves = pd.DataFrame(rows).set_index(["call_set", "classify_threshold"]).sort_index()

# Neither side moves in this sweep, so both denominators have to be constant
# within a call set.
for name in ORDER:
    assert curves.loc[name][["n_query", "n_truth"]].nunique().eq(1).all(), name

TABLES.mkdir(parents=True, exist_ok=True)
curves.to_csv(TABLES / "classification_overlap_curves.csv")

# The geometric claim the subsection rests on, checked rather than asserted in
# prose -- and checked one side at a time, because the two sides do not satisfy
# it equally. Truth is merged at zero padding and is therefore internally
# disjoint within every (sample, chrom, svtype) partition, so no query call can
# reach half of two truth intervals at once and the query-side fan-out is zero at
# 0.5 for every call set. The query side is not disjoint: components built at a
# reciprocal overlap of 0.5 may still overlap one another below that, so a truth
# interval can be split even at the operating threshold.
strict = curves[curves.index.get_level_values("classify_threshold") >= ADOPTED]
assert (strict["n_query_multi"] == 0).all()

focus = curves.loc[FOCUS]


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
    # The adopted value, and the point at which the matching becomes one-to-one.
    ax.axvline(ADOPTED, color="#B0B0B0", linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="x", labelbottom=True, length=3.4)
    ax.tick_params(axis="y", length=3.4)
    if bottom:
        ax.set_xlabel("Classification reciprocal-overlap threshold", labelpad=2)
    ax.text(-0.155, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def legend(ax, handles, loc):
    ax.legend(handles=handles, frameon=False, loc=loc, handlelength=1.8, borderpad=0,
              labelspacing=0.25)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.2), sharex=True)

# (A) The metrics, which decline monotonically: every threshold admits a subset
# of the pairs every looser one admits, so there is nothing here to maximise.
ax = axes[0, 0]
for label, column in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")):
    ax.plot(thresholds, focus[column], color=METRIC_COLORS[label], linewidth=1.4, zorder=3)
ax.set_ylim(0, 1)
dress(ax, "Classification metric", "A", bottom=False)
legend(ax, [Line2D([], [], color=c, linewidth=1.4, label=n) for n, c in METRIC_COLORS.items()],
       "upper right")

# (B) Query-side fan-out. Truth is merged at zero padding and so is internally
# disjoint within each partition; a query call therefore cannot reach half of two
# truth intervals, and every call set is at zero by 0.5 or below.
ax = axes[0, 1]
for name in ORDER:
    ax.plot(thresholds, curves.loc[name]["n_query_multi"], color=SET_COLORS[name],
            linewidth=SET_WIDTHS[name], zorder=3)
# symlog so that zero -- the value the whole panel is about -- has a position.
ax.set_yscale("symlog", linthresh=1, linscale=0.35)
ax.set_ylim(0, 1e3)
dress(ax, "Query calls with $>$1 partner", "B", bottom=False)

# (C) The truth side of the same thing, which does not vanish at 0.5 for every
# call set. Consensus components built at a reciprocal overlap of 0.5 may still
# overlap one another below that, so the 1/3 set is not internally disjoint and
# can split a benchmark interval at the operating threshold.
ax = axes[1, 0]
for name in ORDER:
    ax.plot(thresholds, curves.loc[name]["n_truth_multi"], color=SET_COLORS[name],
            linewidth=SET_WIDTHS[name], zorder=3)
ax.set_yscale("symlog", linthresh=1, linscale=0.35)
ax.set_ylim(0, 2e3)
dress(ax, "Benchmark intervals with $>$1 partner", "C", bottom=True)

# (D) The same decline on all six 30x call sets, to show that the ordering
# between them is not an artefact of where the threshold is put.
ax = axes[1, 1]
for name in ORDER:
    ax.plot(thresholds, curves.loc[name]["f1"], color=SET_COLORS[name],
            linewidth=SET_WIDTHS[name], zorder=3)
ax.set_ylim(0, 0.45)
dress(ax, "F1", "D", bottom=True)
legend(ax, [Line2D([], [], color=SET_COLORS[n], linewidth=SET_WIDTHS[n], label=n) for n in ORDER],
       "upper right")

fig.subplots_adjust(left=0.082, right=0.978, top=0.955, bottom=0.095, wspace=0.21, hspace=0.30)

DEST.mkdir(parents=True, exist_ok=True)
for suffix, dpi in ((".png", 600), (".pdf", None)):
    fig.savefig(DEST / f"classification_overlap{suffix}", dpi=dpi)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
pd.set_option("display.width", 220)
COLS = ["precision", "recall", "f1", "n_true_positive", "n_truth_found", "n_pairs",
        "n_query_multi", "n_truth_multi", "max_query_partners"]
QUOTED = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for name in ORDER:
    block = curves.loc[name]
    print(f"\n=== {name}  (n_query={block['n_query'].iloc[0]}) ===")
    print(block.loc[QUOTED, COLS].round(4).to_string())
print(f"\ntruth intervals: {curves['n_truth'].iloc[0]}")
