"""The 2-of-3 consensus call sets across coverages, against the benchmark and the array.

Where used
----------
Results -> "Performance of 2-of-3 Consensus Call Sets across Coverages":
    Table 10   binary classification of the four coverages and the SNP array
    Figure 11  benchmark recovery as an UpSet plot, and array/sequencing containment
    Figure 12  precision, recall and F1 against CNV size across coverages
    every number quoted in that section

The consensus level is fixed at 2-of-3 by "Consensus Level Selection", and all
four adopted parameters are fixed, so there is no sweep here: one classification
per call set.

Three quantities the section rests on and this script computes rather than
assumes:

  * the benchmark-recovery indicator matrix. Every call set is classified
    against the *same* merged benchmark, so `truth_matched` is a boolean column
    over one fixed universe of intervals and the five columns can be crossed
    directly. This is what the UpSet plot draws, and it replaces the four
    three-way Venn diagrams of the earlier draft.
  * the containment of each recovery set in the other, which is the
    replacement-versus-supplement question stated as two fractions.
  * variant class composition, since the sets drift towards duplications as
    depth falls while the benchmark does not, and that shifts the metrics for a
    reason unrelated to depth.

    pixi run python manuscript/scripts/coverage_performance.py
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
    Classification,
    classify,
    match_topology,
    size_density_curve,
)
from consensuscnv.classification.intervals import IntervalSet
from consensuscnv.classification.pairs import build_candidates

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")
DEST = ROOT / "results" / "coverage_performance"
TABLES = ROOT / "results" / "manuscript"

CALLERS = ("cnvpytor", "delly", "gatk")
BENCHMARKS = ("1000G", "HGSVC3", "ont_vienna")
COVERAGES = ("30x", "6x", "4x", "2x")
ARRAY = "SNP array"
ORDER = (*COVERAGES, ARRAY)
SAMPLES = (
    "HG00096", "HG00171", "HG00268", "HG00513", "HG00731", "HG01596", "HG01890",
    "NA18989", "NA19129", "NA19238", "NA19331", "NA19347", "NA20847",
)

# The adopted parameters, fixed for everything in this script.
SIZE_FLOOR = 1_000
BENCHMARK_PADDING = 0
CONSENSUS_THRESHOLD = 0.5
CLASSIFY_THRESHOLD = 0.5
CONSENSUS_LEVEL = 2

# Blues ramp for the ordinal coverages and mid grey for the array, the same
# assignment Figure 9 already uses.
COLORS = {"30x": "#08519C", "6x": "#3182BD", "4x": "#6BAED6", "2x": "#BDD7E7",
          ARRAY: "#767676"}
# The light end of the ramp disappears against white as a 1 pt line, so the
# curves get a slightly darker variant while the bars keep the ramp.
LINE_COLORS = {**COLORS, "2x": "#9ECAE1"}

SIZE_RANGE = (SIZE_FLOOR, 1_000_000)
BANDWIDTH = 0.15
MIN_EFFECTIVE_COUNT = 20.0
# Combinations to draw in the UpSet matrix. The rest are reported as a remainder
# rather than dropped silently.
MAX_COMBINATIONS = 12


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


def consensus_at(coverage: str) -> IntervalSet:
    """The 2-of-3 consensus set at one coverage.

    Merged once at the default min_sources=1 and then selected on n_sources:
    components come from the edge selection alone and are only afterwards
    dropped, so this reproduces min_sources=2 exactly.
    """
    merged = IntervalSet.from_merged(
        merge_components(
            collect_callsets(read_bed_calls(bed) for bed in bed_paths(ROOT / "out" / f"{coverage}_Coverage", CALLERS)),
            min_reciprocal_overlap=CONSENSUS_THRESHOLD,
        )
    )
    return floored(merged.select(merged.n_sources >= CONSENSUS_LEVEL))


query_sets = {
    **{coverage: consensus_at(coverage) for coverage in COVERAGES},
    # The array was genotyped for the whole 1000 Genomes panel; only the thirteen
    # samples this study sequenced belong in the comparison.
    ARRAY: floored(
        IntervalSet.from_callset(
            collect_callsets(read_bed_calls(str(ROOT / "out" / "SNP_Array" / "bed" / f"{s}.bed")) for s in SAMPLES)
        )
    ),
}

assert len(np.unique(truth.sample_idx)) == len(SAMPLES)
assert all(len(np.unique(query_sets[name].sample_idx)) == len(SAMPLES) for name in ORDER)

DEL_ID, DUP_ID = SVTYPES.intern("DEL"), SVTYPES.intern("DUP")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classified(query: IntervalSet, reference: IntervalSet = truth) -> Classification:
    return classify(
        build_candidates(query, reference), min_reciprocal_overlap=CLASSIFY_THRESHOLD, validate=False
    )


def scored(classification: Classification) -> dict[str, float]:
    """One classification's metrics, plus the recall ceiling.

    `n_true_positive` counts query rows that matched and `n_truth_found` counts
    benchmark rows that were matched. They are equal only when the matching is
    one-to-one, which is a property of the call set rather than of the
    threshold, so both are kept.
    """
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


# One classification per call set, reused by the table, the recovery matrix and
# the size curves. All five share `truth`, which is what makes the columns of
# the recovery matrix comparable row by row.
classifications = {name: classified(query_sets[name]) for name in ORDER}

overall = pd.DataFrame(
    [{"call set": name, **scored(classifications[name])} for name in ORDER]
).set_index("call set")

per_class = pd.DataFrame(
    [
        {
            "call set": name,
            "class": label,
            "fraction of call set": float(np.mean(query_sets[name].svtype_idx == svtype_id)),
            **scored(
                classified(
                    query_sets[name].select(np.where(query_sets[name].svtype_idx == svtype_id)[0]),
                    truth.select(np.where(truth.svtype_idx == svtype_id)[0]),
                )
            ),
        }
        for name in ORDER
        for label, svtype_id in (("DEL", DEL_ID), ("DUP", DUP_ID))
    ]
).set_index(["call set", "class"])

# Where the matching is one-to-one, recall / ceiling is exactly precision: both
# reduce to matched query rows over query rows.
one_to_one = overall[overall["one_to_one"]]
assert np.allclose(one_to_one["recall_over_ceiling"], one_to_one["precision"])

# Candidate pairs never cross variant class, so the two class-restricted
# classifications partition the unrestricted one exactly. This is what lets
# Table 10 carry deletion precision as a decomposition of its precision column
# rather than as a separate measurement.
for column in ("n_query", "n_true_positive", "n_false_positive"):
    totals = per_class[column].groupby("call set").sum().reindex(overall.index)
    assert (totals == overall[column]).all(), column


# --------------------------------------------------------------------------- #
# Benchmark recovery
# --------------------------------------------------------------------------- #
# Column `name` is True at row i if call set `name` recovered benchmark interval
# i. One fixed universe of 24,820 intervals, five indicator columns.
recovery = pd.DataFrame(
    {name: classifications[name].truth_matched for name in ORDER}, columns=list(ORDER)
)
assert len(recovery) == len(truth.starts)
assert all(recovery[name].sum() == overall.loc[name, "n_truth_found"] for name in ORDER)

# The membership pattern of each interval, as a string of five 0/1 characters in
# ORDER. The all-absent pattern is every interval no method recovered.
pattern = recovery.apply(lambda row: "".join("01"[v] for v in row), axis=1)
combinations = pattern.value_counts().drop(index="0" * len(ORDER))
n_recovered = int(combinations.sum())
assert n_recovered == int(recovery.any(axis=1).sum())

drawn = combinations.head(MAX_COMBINATIONS)
remainder = combinations.iloc[MAX_COMBINATIONS:]

# Containment in both directions. The first falls with depth and the second
# rises; where they cross is where sequencing stops subsuming the array.
containment = pd.DataFrame(
    [
        {
            "coverage": coverage,
            "n_shared": (shared := int((recovery[coverage] & recovery[ARRAY]).sum())),
            "n_sequencing_only": int((recovery[coverage] & ~recovery[ARRAY]).sum()),
            "n_array_only": int((~recovery[coverage] & recovery[ARRAY]).sum()),
            "n_neither": int((~recovery[coverage] & ~recovery[ARRAY]).sum()),
            "array_recovery_shared": shared / recovery[ARRAY].sum(),
            "sequencing_recovery_shared": shared / recovery[coverage].sum(),
        }
        for coverage in COVERAGES
    ]
).set_index("coverage")

# The consecutive-coverage nesting the UpSet staircase reports.
nesting = pd.DataFrame(
    [
        {
            "set": lower,
            "contained in": higher,
            "n_shared": int((recovery[lower] & recovery[higher]).sum()),
            "fraction": float((recovery[lower] & recovery[higher]).sum() / recovery[lower].sum()),
        }
        for lower, higher in zip(COVERAGES[1:][::-1], COVERAGES[:-1][::-1], strict=True)
    ]
).set_index("set")

TABLES.mkdir(parents=True, exist_ok=True)
overall.to_csv(TABLES / "coverage_performance.csv")
per_class.to_csv(TABLES / "coverage_performance_by_class.csv")
combinations.rename("n_intervals").to_csv(TABLES / "coverage_recovery_combinations.csv")
containment.to_csv(TABLES / "coverage_array_containment.csv")


# --------------------------------------------------------------------------- #
# Metrics against size
# --------------------------------------------------------------------------- #
# Kernel-smoothed in log10 space, which is where a single bandwidth is
# meaningful across three orders of magnitude of CNV size.
curves = {
    name: size_density_curve(
        classifications[name],
        size_range=SIZE_RANGE,
        bandwidth=BANDWIDTH,
        min_effective_count=MIN_EFFECTIVE_COUNT,
    )
    for name in ORDER
}


# --------------------------------------------------------------------------- #
# Separating composition from depth
# --------------------------------------------------------------------------- #
# Overall precision is the class-wise precisions averaged under the call set's
# own class composition, and that composition drifts hard towards duplications
# as depth falls. Reweighting each coverage's class-wise precisions to the 30x
# composition holds the mixture fixed and leaves only the change in class-wise
# performance, so the gap between the two columns is the part of the fall that
# composition accounts for.
reference_mix = per_class.loc[COVERAGES[0], "fraction of call set"]
composition = pd.DataFrame(
    [
        {
            "call set": name,
            "duplication fraction": per_class.loc[(name, "DUP"), "fraction of call set"],
            "precision": overall.loc[name, "precision"],
            "precision at 30x composition": float(
                sum(reference_mix[label] * per_class.loc[(name, label), "precision"] for label in ("DEL", "DUP"))
            ),
        }
        for name in ORDER
    ]
).set_index("call set")
composition["attributable to composition"] = (
    composition["precision at 30x composition"] - composition["precision"]
)
# The reweighting is exact at the reference coverage by construction.
assert np.isclose(composition.loc[COVERAGES[0], "attributable to composition"], 0.0, atol=1e-9)
composition.to_csv(TABLES / "coverage_composition_effect.csv")


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

PRESENT_DOT = "#333333"
ABSENT_DOT = "#DCDCDC"
SHADE = "#F2F2F2"


def panel(ax, letter: str, x: float = -0.20) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.4)
    ax.text(x, 1.03, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def save(fig, stem: str) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in ((".png", 600), (".pdf", None)):
        fig.savefig(DEST / f"{stem}{suffix}", dpi=dpi)


# --------------------------------------------------------------------------- #
# Figure 11 -- benchmark recovery
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(7.09, 3.05))
left, right = fig.subfigures(1, 2, width_ratios=[2.25, 1.0])

# (A) UpSet. Three axes on one grid: intersection sizes on top, the membership
# matrix below it sharing the x axis, and each set's total to the left of the
# matrix sharing the y axis. Only intervals recovered by at least one method are
# shown; the rest are stated in the caption.
# The set labels sit in the gap between the totals and the matrix, so `wspace`
# has to clear the longest of them.
grid = left.add_gridspec(2, 2, width_ratios=[0.34, 1.0], height_ratios=[1.0, 0.62],
                         wspace=0.32, hspace=0.06,
                         left=0.03, right=0.99, top=0.90, bottom=0.10)
ax_bars = left.add_subplot(grid[0, 1])
ax_matrix = left.add_subplot(grid[1, 1], sharex=ax_bars)
ax_totals = left.add_subplot(grid[1, 0], sharey=ax_matrix)

positions = np.arange(len(drawn))
ax_bars.bar(positions, drawn.to_numpy(), width=0.62, color=PRESENT_DOT, zorder=2)
for x, size in zip(positions, drawn.to_numpy(), strict=True):
    ax_bars.text(x, size + drawn.max() * 0.03, f"{size:,}", ha="center", va="bottom", fontsize=5.5)
ax_bars.set_ylim(0, drawn.max() * 1.20)
ax_bars.set_ylabel("Benchmark intervals", labelpad=2)
ax_bars.tick_params(labelbottom=False)
panel(ax_bars, "A", x=-0.42)

for row in range(len(ORDER)):
    if row % 2 == 0:
        ax_matrix.axhspan(row - 0.5, row + 0.5, color=SHADE, zorder=0, linewidth=0)
for x, key in zip(positions, drawn.index, strict=True):
    members = [row for row, flag in enumerate(key) if flag == "1"]
    ax_matrix.plot([x] * len(ORDER), range(len(ORDER)), "o", color=ABSENT_DOT,
                   markersize=4.2, zorder=2)
    ax_matrix.plot([x, x], [min(members), max(members)], color=PRESENT_DOT,
                   linewidth=1.1, zorder=3)
    ax_matrix.plot([x] * len(members), members, "o", color=PRESENT_DOT,
                   markersize=4.2, zorder=4)
ax_matrix.set_xlim(-0.7, len(drawn) - 0.3)
ax_matrix.set_ylim(len(ORDER) - 0.5, -0.5)
ax_matrix.set_yticks(range(len(ORDER)), ORDER)
ax_matrix.set_xticks([])
for side in ("top", "right", "bottom", "left"):
    ax_matrix.spines[side].set_visible(False)
ax_matrix.tick_params(length=0)

ax_totals.barh(range(len(ORDER)), [recovery[name].sum() for name in ORDER],
               height=0.62, color=[COLORS[name] for name in ORDER],
               edgecolor="#000000", linewidth=0.4, zorder=2)
ax_totals.invert_xaxis()
ax_totals.set_xlabel("Recovered", labelpad=2)
ax_totals.tick_params(left=False, labelleft=False, length=3.0)
ax_totals.set_xticks([0, 2000, 4000], ["0", "2k", "4k"])
for side in ("top", "right", "left"):
    ax_totals.spines[side].set_visible(False)

# (B) The two containment fractions. They answer whether sequencing subsumes the
# array or discovers a different population, and they cross between 4x and 2x.
ax = right.subplots()
right.subplots_adjust(left=0.24, right=0.98, top=0.90, bottom=0.185)
depths = np.arange(len(COVERAGES))
for column, color, marker in (("array_recovery_shared", "#767676", "o"),
                              ("sequencing_recovery_shared", "#08519C", "s")):
    ax.plot(depths, containment[column], color=color, marker=marker, markersize=4,
            linewidth=1.2, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
ax.set_xticks(depths, COVERAGES)
ax.set_xlim(-0.3, len(COVERAGES) - 0.7)
ax.set_ylim(0, 1)
ax.set_xlabel("Sequencing coverage", labelpad=2)
ax.set_ylabel("Shared with the other method", labelpad=2)
ax.legend(handles=[
    Line2D([], [], color="#767676", marker="o", markersize=3.6, linewidth=1.2,
           label="of array recoveries"),
    Line2D([], [], color="#08519C", marker="s", markersize=3.6, linewidth=1.2,
           label="of sequencing recoveries"),
], frameon=False, loc="upper center", handlelength=1.4, borderpad=0,
    labelspacing=0.22, handletextpad=0.4)
panel(ax, "B", x=-0.30)

save(fig, "benchmark_recovery")


# --------------------------------------------------------------------------- #
# Figure 12 -- metrics against size
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.45))
for ax, (attribute, label) in zip(
    axes, (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")), strict=True
):
    for name in ORDER:
        ax.plot(curves[name].sizes, getattr(curves[name], attribute),
                color=LINE_COLORS[name], linewidth=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(*SIZE_RANGE)
    ax.set_xticks([1e3, 1e4, 1e5, 1e6], ["1 kb", "10 kb", "100 kb", "1 Mb"])
    ax.set_xlabel("CNV size", labelpad=2)
    ax.set_ylabel(label, labelpad=2)
    ax.set_ylim(0, None)
for ax, letter in zip(axes, "ABC", strict=True):
    panel(ax, letter)
axes[0].legend(handles=[Line2D([], [], color=LINE_COLORS[n], linewidth=1.2, label=n)
                        for n in ORDER],
               frameon=False, loc="upper right", handlelength=1.2, borderpad=0,
               labelspacing=0.22, handletextpad=0.4)
fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.165, wspace=0.36)
save(fig, "coverage_size_metrics")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
pd.set_option("display.width", 200, "display.max_columns", 40)

print("Benchmark intervals above the size floor:", len(truth.starts))
print("\n--- Table 10: 2-of-3 consensus across coverages, and the array ---")
table10 = overall[["n_query", "n_true_positive", "n_false_positive", "precision",
                   "recall", "recall_ceiling", "f1"]].copy()
table10.insert(4, "precision (DEL)", per_class.xs("DEL", level="class")["precision"])
print(table10.round(3).to_string())
print("\nfull:")
print(overall.round(4).to_string())

print("\n--- Variant class ---")
print(per_class.round(4).to_string())

print("\n--- Benchmark recovery ---")
print(f"recovered by at least one method: {n_recovered:,} of {len(truth.starts):,} "
      f"({n_recovered / len(truth.starts):.1%}); by none: {len(truth.starts) - n_recovered:,}")
print(f"nonzero combinations: {len(combinations)} of {2 ** len(ORDER) - 1}; "
      f"the {len(drawn)} drawn hold {drawn.sum():,} ({drawn.sum() / n_recovered:.1%}), "
      f"the remaining {len(remainder)} hold {remainder.sum():,}")
print("\ncombination sizes (columns in ORDER):")
print(combinations.to_string())

print("\n--- Nesting across consecutive coverages ---")
print(nesting.round(4).to_string())

print("\n--- Sequencing against the array ---")
print(containment.round(4).to_string())

print("\n--- Composition against depth ---")
print(composition.round(4).to_string())

print("\n--- Metrics against size ---")
CHECKPOINTS = (2_000, 5_000, 10_000, 50_000, 200_000)
at_sizes = pd.DataFrame(
    [
        {
            "call set": name,
            "size": size,
            **{
                attribute: float(np.interp(np.log10(size), np.log10(curves[name].sizes),
                                           getattr(curves[name], attribute)))
                for attribute in ("precision", "recall", "f1")
            },
        }
        for name in ORDER
        for size in CHECKPOINTS
    ]
).set_index(["call set", "size"])
at_sizes.to_csv(TABLES / "coverage_size_checkpoints.csv")
print(at_sizes.round(4).to_string())

for name in ORDER:
    curve = curves[name]
    finite = np.isfinite(curve.f1)
    if not finite.any():
        print(f"{name:>10s}: no size supported at min_effective_count={MIN_EFFECTIVE_COUNT}")
        continue
    peak = int(np.nanargmax(np.where(finite, curve.f1, -np.inf)))
    print(f"{name:>10s}: F1 peaks at {curve.f1[peak]:.3f} near {curve.sizes[peak]:,.0f} bp; "
          f"supported over {curve.sizes[finite].min():,.0f}-{curve.sizes[finite].max():,.0f} bp")
