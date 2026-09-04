"""Performance of the 30x call sets against the merged benchmark, at the adopted parameters.

Where used
----------
Results -> "Consensus Level Selection":
    Table 9    binary classification of the six 30x call sets and the SNP array
    Figure 10  the precision/recall plane, the agreement strata, and F1 against size
    every number quoted in that section

The question the section answers is which consensus level to carry into the
coverage comparison. Everything is at 30x and at the four adopted parameters, so
there is no sweep here: one classification per call set.

Three quantities the section rests on and this script computes rather than
assumes:

  * the recall ceiling, n_query / n_truth capped at one. Recall is scored
    against the whole merged benchmark, which is larger than every query set, so
    a set's recall is bounded by how many calls it contains. Where the matching
    is one-to-one, recall divided by that ceiling is exactly precision.
  * precision by the number of callers that reported a component, taken over
    *exactly* k callers rather than at least k, which is what separates the
    caller-private stratum from the agreeing one.
  * the same metrics restricted to deletions and to duplications, since the
    benchmark holds two orders of magnitude more of the former.

    pixi run python manuscript/scripts/consensus_levels.py
"""

import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from consensuscnv.callsets import collect_callsets, merge_components, read_bed_calls
from consensuscnv.callsets.registry import SVTYPES
from consensuscnv.classification.classify import (
    classify,
    match_topology,
    size_density_curve,
)
from consensuscnv.classification.intervals import IntervalSet
from consensuscnv.classification.pairs import build_candidates

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")
DEST = ROOT / "results" / "consensus_levels"
TABLES = ROOT / "results" / "manuscript"

CALLERS = ("cnvpytor", "delly", "gatk")
LABELS = {"cnvpytor": "CNVpytor", "delly": "Delly", "gatk": "GATK-gCNV"}
BENCHMARKS = ("1000G", "HGSVC3", "ont_vienna")
CONSENSUS_LEVELS = (1, 2, 3)
SAMPLES = (
    "HG00096", "HG00171", "HG00268", "HG00513", "HG00731", "HG01596", "HG01890",
    "NA18989", "NA19129", "NA19238", "NA19331", "NA19347", "NA20847",
)

# The adopted parameters, fixed for everything in this script.
SIZE_FLOOR = 1_000
BENCHMARK_PADDING = 0
CONSENSUS_THRESHOLD = 0.5
CLASSIFY_THRESHOLD = 0.5

# Same assignment as every other figure: Okabe-Ito for the callers, a Purples
# ramp for the ordinal consensus levels, mid grey for the array.
SET_COLORS = {
    "CNVpytor": "#0072B2", "Delly": "#D55E00", "GATK-gCNV": "#009E73",
    "1/3": "#9E9AC8", "2/3": "#6A51A3", "3/3": "#3F007D", "SNP array": "#767676",
}
STRATUM_COLORS = {1: "#9E9AC8", 2: "#6A51A3", 3: "#3F007D"}
ORDER = ["CNVpytor", "Delly", "GATK-gCNV", "1/3", "2/3", "3/3", "SNP array"]
FOCUS = "2/3"


def bed_paths(root: Path, subdirs: tuple[str, ...]) -> list[str]:
    """Every per-sample BED under the named subdirectories of `root`.

    Pinned to an explicit tuple rather than a wildcard: consensus output is
    destined to land beside the per-caller directories, and a glob would read it
    back in as a fourth source.
    """
    return [bed for sub in subdirs for bed in sorted(glob.glob(str(root / sub / "*.bed")))]


def floored(intervals: IntervalSet) -> IntervalSet:
    return intervals.filter_by_size(min_size=SIZE_FLOOR)


# --------------------------------------------------------------------------- #
# Call sets
# --------------------------------------------------------------------------- #
truth = floored(
    IntervalSet.from_merged(
        merge_components(
            collect_callsets(read_bed_calls(bed) for bed in bed_paths(ROOT / "out" / "benchmark", BENCHMARKS)),
            max_padding=BENCHMARK_PADDING,
        )
    )
)

coverage_dir = ROOT / "out" / "30x_Coverage"
# Merged once at the default min_sources=1: components come from the edge
# selection alone and are only afterwards dropped, so selecting on n_sources
# reproduces any consensus level exactly.
consensus = IntervalSet.from_merged(
    merge_components(
        collect_callsets(read_bed_calls(bed) for bed in bed_paths(coverage_dir, CALLERS)),
        min_reciprocal_overlap=CONSENSUS_THRESHOLD,
    )
)
query_sets = {
    **{
        LABELS[caller]: floored(
            IntervalSet.from_callset(
                collect_callsets(read_bed_calls(bed) for bed in bed_paths(coverage_dir, (caller,)))
            )
        )
        for caller in CALLERS
    },
    **{f"{k}/3": floored(consensus.select(consensus.n_sources >= k)) for k in CONSENSUS_LEVELS},
    # The array was genotyped for the whole 1000 Genomes panel; only the thirteen
    # samples this study sequenced belong in the comparison.
    "SNP array": floored(
        IntervalSet.from_callset(
            collect_callsets(read_bed_calls(str(ROOT / "out" / "SNP_Array" / "bed" / f"{s}.bed")) for s in SAMPLES)
        )
    ),
}
# Strata are *exactly* k callers, not at least k: the k=1 stratum is the
# caller-private population the agreement null model failed to account for.
strata = {k: floored(consensus.select(consensus.n_sources == k)) for k in CONSENSUS_LEVELS}

assert len(np.unique(truth.sample_idx)) == len(SAMPLES)
assert len(np.unique(query_sets["SNP array"].sample_idx)) == len(SAMPLES)

DEL_ID, DUP_ID = SVTYPES.intern("DEL"), SVTYPES.intern("DUP")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def scored(query: IntervalSet, reference: IntervalSet = truth) -> dict[str, float]:
    """One classification at the adopted threshold, plus the recall ceiling.

    `n_true_positive` counts query rows that matched and `n_truth_found` counts
    benchmark rows that were matched. They are equal only when the matching is
    one-to-one, which is a property of the call set rather than of the
    threshold, so both are kept.
    """
    classification = classify(
        build_candidates(query, reference), min_reciprocal_overlap=CLASSIFY_THRESHOLD, validate=False
    )
    summary = classification.summary()
    ceiling = min(1.0, summary.n_query / summary.n_truth) if summary.n_truth else float("nan")
    return {
        "n_query": summary.n_query,
        "n_truth": summary.n_truth,
        "n_true_positive": summary.n_true_positive,
        "n_false_positive": summary.n_false_positive,
        "n_truth_found": summary.n_truth_found,
        "precision": summary.precision,
        "recall": summary.recall,
        "recall_ceiling": ceiling,
        "recall_over_ceiling": summary.recall / ceiling if ceiling else float("nan"),
        "f1": summary.f1,
        "one_to_one": match_topology(classification).is_one_to_one,
    }


def by_class(query: IntervalSet, svtype_id: int) -> dict[str, float]:
    """The same scoring with both sides restricted to one variant class."""
    return scored(
        query.select(np.where(query.svtype_idx == svtype_id)[0]),
        truth.select(np.where(truth.svtype_idx == svtype_id)[0]),
    )


overall = pd.DataFrame(
    [{"call set": name, **scored(query_sets[name])} for name in ORDER]
).set_index("call set")

per_class = pd.DataFrame(
    [
        {"call set": name, "class": label, **by_class(query_sets[name], svtype_id)}
        for name in ORDER
        for label, svtype_id in (("DEL", DEL_ID), ("DUP", DUP_ID))
    ]
).set_index(["call set", "class"])

by_stratum = pd.DataFrame(
    [{"callers": k, **scored(strata[k])} for k in CONSENSUS_LEVELS]
).set_index("callers")

# Where the matching is one-to-one, recall / ceiling is exactly precision: both
# reduce to matched query rows over query rows. The recall column therefore
# separates the call sets by size, not by detection, and the section says so.
one_to_one = overall[overall["one_to_one"]]
assert np.allclose(one_to_one["recall_over_ceiling"], one_to_one["precision"])

# The strata partition the 1/3 set, so their counts have to add back up to it.
assert by_stratum["n_query"].sum() == overall.loc["1/3", "n_query"]
assert by_stratum["n_false_positive"].sum() == overall.loc["1/3", "n_false_positive"]

# Candidate pairs never cross variant class, so the two class-restricted
# classifications partition the unrestricted one exactly. This is what lets
# Table 9 carry deletion precision as a decomposition of its precision column
# rather than as a separate measurement.
for column in ("n_query", "n_true_positive", "n_false_positive"):
    totals = per_class[column].groupby("call set").sum().reindex(overall.index)
    assert (totals == overall[column]).all(), column

TABLES.mkdir(parents=True, exist_ok=True)
overall.to_csv(TABLES / "consensus_levels_30x.csv")
per_class.to_csv(TABLES / "consensus_levels_30x_by_class.csv")
by_stratum.to_csv(TABLES / "consensus_levels_agreement_strata.csv")


# --------------------------------------------------------------------------- #
# Metrics against size
# --------------------------------------------------------------------------- #
# Kernel-smoothed in log10 space, which is where a single bandwidth is
# meaningful across three orders of magnitude of CNV size.
SIZE_RANGE = (SIZE_FLOOR, 1_000_000)
curves = {
    name: size_density_curve(
        classify(
            build_candidates(query_sets[name], truth),
            min_reciprocal_overlap=CLASSIFY_THRESHOLD,
            validate=False,
        ),
        size_range=SIZE_RANGE,
        bandwidth=0.15,
        min_effective_count=50.0,
    )
    for name in ORDER
}


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


def panel(ax, letter: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.4)
    ax.text(-0.20, 1.03, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.45))

# (A) The operating point of every call set in one plane, with the recall
# ceiling drawn as the bar each point sits on. The distance from a point to the
# right end of its bar is the recall the set forfeits by being smaller than the
# benchmark, which is not a detection failure.
ax = axes[0]
grid = np.linspace(0.001, 1, 400)
precision_grid, recall_grid = np.meshgrid(grid, grid, indexing="ij")
f1_grid = 2 * precision_grid * recall_grid / (precision_grid + recall_grid)
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
contours = ax.contour(recall_grid, precision_grid, f1_grid, levels=LEVELS,
                      colors="#C8C8C8", linewidths=0.5, zorder=1)
# Label each contour where it crosses precision == recall, which is the one
# point on it guaranteed to be inside the axes.
ax.clabel(contours, fmt="%.1f", fontsize=5.5, inline=True, inline_spacing=2,
          manual=[(level, level) for level in LEVELS])
for name in ORDER:
    row = overall.loc[name]
    ax.plot([row["recall"], row["recall_ceiling"]], [row["precision"]] * 2,
            color=SET_COLORS[name], linewidth=0.7, alpha=0.55, zorder=2)
    ax.plot(row["recall_ceiling"], row["precision"], marker="|", markersize=4,
            color=SET_COLORS[name], zorder=2)
    ax.plot(row["recall"], row["precision"], marker="o", markersize=4.2,
            color=SET_COLORS[name], markeredgecolor="white", markeredgewidth=0.5, zorder=3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Recall", labelpad=2)
ax.set_ylabel("Precision", labelpad=2)
panel(ax, "A")
ax.legend(handles=[Line2D([], [], color=SET_COLORS[n], marker="o", markersize=3.6,
                          linestyle="none", label=n) for n in ORDER],
          frameon=False, loc="upper right", handlelength=1.0, borderpad=0,
          labelspacing=0.22, handletextpad=0.4)

# (B) Precision by the number of callers that reported a component. Exactly one
# caller is the private stratum, which is where the false positives are.
ax = axes[1]
ax.bar(CONSENSUS_LEVELS, by_stratum["precision"], width=0.62,
       color=[STRATUM_COLORS[k] for k in CONSENSUS_LEVELS], zorder=2)
for k in CONSENSUS_LEVELS:
    ax.text(k, by_stratum.loc[k, "precision"] + 0.025,
            f"{by_stratum.loc[k, 'precision']:.2f}\nn = {by_stratum.loc[k, 'n_query']:,}",
            ha="center", va="bottom", fontsize=6)
ax.set_xticks(list(CONSENSUS_LEVELS))
ax.set_xlabel("Callers reporting the component", labelpad=2)
ax.set_ylabel("Precision", labelpad=2)
ax.set_ylim(0, 1.22)
ax.set_yticks(np.arange(0, 1.01, 0.2))
panel(ax, "B")

# (C) F1 against CNV size. The aggregate ranking in panel A is a single number
# per set; this is whether that ranking holds across the size domain.
ax = axes[2]
for name in ORDER:
    curve = curves[name]
    ax.plot(curve.sizes, curve.f1, color=SET_COLORS[name],
            linewidth=1.5 if name.endswith("/3") else 1.0, zorder=3)
ax.set_xscale("log")
ax.set_xlim(*SIZE_RANGE)
ax.set_xticks([1e3, 1e4, 1e5, 1e6])
ax.set_xticklabels(["1 kb", "10 kb", "100 kb", "1 Mb"])
ax.set_xlabel("CNV size", labelpad=2)
ax.set_ylabel("F1", labelpad=2)
ax.set_ylim(0, None)
panel(ax, "C")

fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.165, wspace=0.36)

DEST.mkdir(parents=True, exist_ok=True)
for suffix, dpi in ((".png", 600), (".pdf", None)):
    fig.savefig(DEST / f"consensus_levels{suffix}", dpi=dpi)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
pd.set_option("display.width", 240)
print("\n=== 30x call sets vs. merged benchmark, adopted parameters ===")
print(overall.round(4).to_string())
print("\n=== by variant class ===")
print(per_class.round(4).to_string())
print("\n=== agreement strata (exactly k callers) ===")
print(by_stratum.round(4).to_string())
print("\n=== F1 against size ===")
for name in ORDER:
    curve = curves[name]
    peak = int(np.nanargmax(curve.f1))
    print(f"{name:>10}  peak F1 {curve.f1[peak]:.3f} at {curve.sizes[peak]:,.0f} bp")

# Where the 2/3 set overtakes the 1/3 set, which is the only crossing the
# section quotes from panel C.
gap = curves["2/3"].f1 - curves["1/3"].f1
crossings = np.where(np.diff(np.signbit(gap)))[0]
print("2/3 overtakes 1/3 at:", [f"{curves['2/3'].sizes[i]:,.0f} bp" for i in crossings])

print(f"\nbenchmark: {len(truth):,} intervals "
      f"({int(np.sum(truth.svtype_idx == DEL_ID)):,} DEL, {int(np.sum(truth.svtype_idx == DUP_ID)):,} DUP)")
