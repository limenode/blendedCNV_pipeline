"""Joint four-factor grid: Sobol' sensitivity indices and the precision-recall Pareto front.

Where used
----------
Results -> "Variance-based sensitivity analysis":
    Figure (sensitivity)  four-panel summary of the joint grid
    every number quoted in that section
Supplementals:
    results/manuscript/sensitivity_indices.tsv         full decomposition, 2/3 consensus
    results/manuscript/sensitivity_indices_wide.tsv    the same on an over-wide grid
    results/manuscript/pareto_front.csv                the non-dominated settings

The three preceding subsections vary one parameter at a time. That reading is
complete only if each parameter's effect is the same wherever the other three
are held, which is a claim about the shape of the metric field and not
something a profile can test. Here all four are varied together over the full
factorial grid, and two questions are asked of the result:

    how much of the variance does each parameter carry, alone and jointly
        (Sobol' first- and total-order indices; the sum of the first-order
        indices is the R^2 of the additive model, i.e. of the profiles)
    which settings are not beaten on both precision and recall at once
        (the Pareto front)

The front matters here because precision and recall live on very different
scales in this comparison -- precision is several times recall throughout -- so
F1 is nearly a monotone function of recall alone and a single-objective optimum
would hide the trade rather than resolve it.

    pixi run python manuscript/scripts/sensitivity.py
"""

import glob
import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from consensuscnv.analysis.sobol import decompose, sobol_indices, write_sensitivity_tsv
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
FOCUS_LEVEL = 2             # the adopted call set; the one reported in the text

# Axis order is the standing four-parameter order of the paper.
FACTORS = ("benchmark padding", "size floor", "consensus overlap", "classification overlap")
PADDING = np.array([0, 10, 25, 50, 100, 200, 400, 700, 1000])
FLOOR = np.array([0, 250, 500, 1000, 2000, 5000, 10000])
CONSENSUS = np.round(np.arange(0.05, 1.00, 0.05), 2)
CLASSIFY = np.round(np.arange(0.05, 1.00, 0.05), 2)

# Sobol' indices are variance ratios with respect to a distribution over the
# inputs, so they describe the region examined and nothing outside it. The wide
# grid deliberately runs the two size parameters out to 10^6 bp, far past any
# defensible value, to show how much of the reported attribution is a statement
# about the pipeline and how much is a statement about the ranges.
WIDE_PADDING = np.array([0, 100, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000])
WIDE_FLOOR = np.array([0, 1_000, 10_000, 30_000, 100_000, 300_000, 1_000_000])

ADOPTED = {"benchmark padding": 0, "size floor": 1_000,
           "consensus overlap": 0.5, "classification overlap": 0.5}

# Okabe-Ito, as everywhere else in the paper: unordered categories.
METRIC_COLORS = {"precision": "#0072B2", "recall": "#D55E00", "f1": "#009E73"}
FACTOR_COLORS = {
    "benchmark padding": "#0072B2",
    "size floor": "#D55E00",
    "consensus overlap": "#009E73",
    "classification overlap": "#CC79A7",
}
GREY, MIDGREY = "#BDBDBD", "#767676"


def bed_paths(root: Path, subdirs: tuple[str, ...]) -> list[str]:
    """Every per-sample BED under the named subdirectories of `root`.

    Pinned to an explicit tuple rather than a wildcard: consensus output is
    destined to land beside the per-caller directories, and a glob would read it
    back in as a fourth source.
    """
    return [bed for sub in subdirs for bed in sorted(glob.glob(str(root / sub / "*.bed")))]


# --------------------------------------------------------------------------- #
# Call sets. Both graphs are built once and every grid point is a filter on them.
# --------------------------------------------------------------------------- #
benchmark = collect_callsets(read_bed_calls(bed) for bed in bed_paths(ROOT / "out" / "benchmark", BENCHMARKS))
raw = collect_callsets(read_bed_calls(bed) for bed in bed_paths(ROOT / "out" / "30x_Coverage", CALLERS))


def sweep(padding: np.ndarray, floor: np.ndarray, levels: tuple[int, ...]) -> dict[int, dict[str, np.ndarray]]:
    """Evaluate the full factorial grid, one field per metric per consensus level.

    Every field has shape (padding, floor, consensus, classification), the
    standing parameter order.
    """
    truth = {
        (i, j): IntervalSet.from_merged(merged).filter_by_size(min_size=int(f))
        for i, merged in ((i, merge_components(benchmark, max_padding=int(p))) for i, p in enumerate(padding))
        for j, f in enumerate(floor)
    }
    query_merged = {
        k: IntervalSet.from_merged(merge_components(raw, min_reciprocal_overlap=float(c)))
        for k, c in enumerate(CONSENSUS)
    }

    shape = (len(padding), len(floor), len(CONSENSUS), len(CLASSIFY))
    fields = {
        level: {name: np.empty(shape) for name in ("precision", "recall", "f1", "n_query", "n_truth")}
        for level in levels
    }

    for (i, _), (j, f), (k, _) in itertools.product(
        enumerate(padding), enumerate(floor), enumerate(CONSENSUS)
    ):
        intervals = query_merged[k]
        for level in levels:
            # min_sources is a post-hoc component filter, so selecting
            # n_sources >= level off the single unfiltered merge is exactly what
            # min_sources=level returns.
            keep = np.flatnonzero((intervals.n_sources >= level) & (intervals.lengths >= int(f)))
            candidates = build_candidates(intervals.select(keep), truth[(i, j)])
            for m, threshold in enumerate(CLASSIFY):
                s = classify(candidates, min_reciprocal_overlap=float(threshold), validate=False).summary()
                cell = fields[level]
                cell["precision"][i, j, k, m] = s.precision
                cell["recall"][i, j, k, m] = s.recall
                cell["f1"][i, j, k, m] = s.f1
                cell["n_query"][i, j, k, m] = s.n_query
                cell["n_truth"][i, j, k, m] = s.n_truth
    return fields


fields = sweep(PADDING, FLOOR, CONSENSUS_LEVELS)
focus = fields[FOCUS_LEVEL]

# The two call sets depend on some axes and not others. If either of these
# fails, a metric is moving for a reason the attribution below cannot see.
assert np.ptp(focus["n_truth"], axis=(2, 3)).max() == 0, "truth set moved with a query-side parameter"
assert np.ptp(focus["n_query"], axis=(0, 3)).max() == 0, "query set moved with a truth-side parameter"
assert np.isfinite(np.stack([focus[m] for m in ("precision", "recall", "f1")])).all()


# --------------------------------------------------------------------------- #
# Sensitivity indices
# --------------------------------------------------------------------------- #
TABLES.mkdir(parents=True, exist_ok=True)

indices = {name: sobol_indices(focus[name], FACTORS) for name in ("precision", "recall", "f1")}
write_sensitivity_tsv(TABLES / "sensitivity_indices.tsv", indices)

# The additive model is the profiles of the preceding sections written as one
# object: a grand mean plus a single curve per parameter. Its R^2 is the share
# of the joint field those profiles reproduce.
models = {name: decompose(focus[name]) for name in ("precision", "recall", "f1")}

# Are the same parameters dominant at the other two consensus levels?
by_level = pd.DataFrame([
    {
        "call_set": f"{level}/3",
        "metric": name,
        "factor": factor,
        "first_order": float(idx.first_order[a]),
        "total_order": float(idx.total_order[a]),
        "additive_r2": float(idx.additive_fraction),
    }
    for level in CONSENSUS_LEVELS
    for name in ("precision", "recall", "f1")
    for idx in [sobol_indices(fields[level][name], FACTORS, include_interactions=False)]
    for a, factor in enumerate(FACTORS)
]).set_index(["call_set", "metric", "factor"]).sort_index()
by_level.to_csv(TABLES / "sensitivity_indices_by_level.csv")

# The same analysis on the over-wide grid, for the ranges caveat.
wide = sweep(WIDE_PADDING, WIDE_FLOOR, (FOCUS_LEVEL,))[FOCUS_LEVEL]
wide_indices = {name: sobol_indices(wide[name], FACTORS) for name in ("precision", "recall", "f1")}
write_sensitivity_tsv(TABLES / "sensitivity_indices_wide.tsv", wide_indices)


# --------------------------------------------------------------------------- #
# The grid itself, and its Pareto front
# --------------------------------------------------------------------------- #
grid = pd.DataFrame({
    "benchmark padding": np.repeat(PADDING, len(FLOOR) * len(CONSENSUS) * len(CLASSIFY)),
    "size floor": np.tile(np.repeat(FLOOR, len(CONSENSUS) * len(CLASSIFY)), len(PADDING)),
    "consensus overlap": np.tile(np.repeat(CONSENSUS, len(CLASSIFY)), len(PADDING) * len(FLOOR)),
    "classification overlap": np.tile(CLASSIFY, len(PADDING) * len(FLOOR) * len(CONSENSUS)),
    **{name: focus[name].ravel() for name in ("precision", "recall", "f1", "n_query", "n_truth")},
})
grid.to_csv(TABLES / f"sensitivity_grid_{FOCUS_LEVEL}of3.csv", index=False)


def pareto_mask(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """Non-dominated settings: no other setting is >= on both and > on one.

    Sweeping in order of decreasing recall, a setting joins the front when its
    precision exceeds every precision seen so far; ties in recall are handled by
    the secondary sort, so of two settings with identical coordinates only one
    is kept.
    """
    order = np.lexsort((-precision, -recall))
    keep, best = [], -np.inf
    for idx in order:
        if precision[idx] > best:
            keep.append(idx)
            best = precision[idx]
    mask = np.zeros(precision.size, dtype=bool)
    mask[keep] = True
    return mask


def pareto_front(frame: pd.DataFrame) -> pd.DataFrame:
    keep = pareto_mask(frame["precision"].to_numpy(), frame["recall"].to_numpy())
    return frame[keep].sort_values("recall", ascending=False)


front = pareto_front(grid)
front.to_csv(TABLES / "pareto_front.csv", index=False)

# The classification threshold is not a performance parameter: it defines what a
# correct call is, so a setting that scores better only because it credits more
# loosely is not a setting that does better. Both metrics decline in it
# monotonically, so it is dominated everywhere and the unconditional front is
# uninformative about it. The decision-relevant fronts hold it at the adopted
# value, and then also hold the floor at the value the resolution argument fixes.
at_half = grid[grid["classification overlap"] == ADOPTED["classification overlap"]]
front_half = pareto_front(at_half)
front_half.to_csv(TABLES / "pareto_front_at_adopted_classification.csv", index=False)

at_floor = at_half[at_half["size floor"] == ADOPTED["size floor"]]
front_floor = pareto_front(at_floor)
front_floor.to_csv(TABLES / "pareto_front_at_adopted_floor.csv", index=False)

adopted = grid.loc[
    (grid["benchmark padding"] == ADOPTED["benchmark padding"])
    & (grid["size floor"] == ADOPTED["size floor"])
    & (grid["consensus overlap"] == ADOPTED["consensus overlap"])
    & (grid["classification overlap"] == ADOPTED["classification overlap"])
].iloc[0]
best_f1 = grid.loc[grid["f1"].idxmax()]
def dominators(frame: pd.DataFrame, point: pd.Series) -> pd.DataFrame:
    """Settings that beat `point` on one metric without losing on the other."""
    return frame[
        (frame["precision"] >= point["precision"])
        & (frame["recall"] >= point["recall"])
        & ((frame["precision"] > point["precision"]) | (frame["recall"] > point["recall"]))
    ]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
print(f"grid: {grid.shape[0]} settings per consensus level")
for name, idx in indices.items():
    print(f"{name}: {idx}")
    print(f"    additive R^2 {models[name].r_squared:.4f}, "
          f"largest interaction {max(idx.interactions.items(), key=lambda kv: kv[1])}")
print("wide grid:")
for name, idx in wide_indices.items():
    print(f"    {name}: {idx}")
for label, frame, this_front in (
    ("whole grid", grid, front),
    ("at classification 0.5", at_half, front_half),
    ("at classification 0.5 and floor 1 kb", at_floor, front_floor),
):
    print(f"front, {label}: {len(this_front)} of {len(frame)} settings")
    print(f"    recall {this_front['recall'].min():.4f}-{this_front['recall'].max():.4f}, "
          f"precision {this_front['precision'].min():.4f}-{this_front['precision'].max():.4f}")
    for factor in FACTORS:
        values = sorted(this_front[factor].unique())
        print(f"    {factor}: {[f'{v:g}' for v in values]}")
    beaten = dominators(frame, adopted)
    print(f"    adopted setting dominated by {len(beaten)} of these settings")
print(f"adopted: precision {adopted['precision']:.4f} recall {adopted['recall']:.4f} f1 {adopted['f1']:.4f}")
print("by consensus level, first-order indices for f1:")
print(by_level.xs("f1", level="metric")["first_order"].unstack().round(3))
print("additive R^2 by level and metric:")
print(by_level["additive_r2"].groupby(level=["call_set", "metric"]).first().unstack().round(3))
print(f"best f1: {best_f1['f1']:.4f} at "
      + ", ".join(f"{f} {best_f1[f]:g}" for f in FACTORS)
      + f" (precision {best_f1['precision']:.4f}, recall {best_f1['recall']:.4f})")


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

SHORT = {"benchmark padding": "padding", "size floor": "floor",
         "consensus overlap": "consensus", "classification overlap": "classification"}


def letter(ax, mark: str) -> None:
    ax.text(-0.155, 1.02, mark, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.2))
(ax_a, ax_b), (ax_c, ax_d) = axes

# (A) first- and total-order indices, one group per factor plus the residue.
groups = [*FACTORS, "interactions"]
x = np.arange(len(groups))
width = 0.26
for offset, (name, idx) in zip((-width, 0.0, width), indices.items()):
    first = [*idx.first_order, idx.interaction_fraction]
    total = [*idx.total_order, idx.interaction_fraction]
    ax_a.bar(x + offset, total, width, color=METRIC_COLORS[name], alpha=0.28, linewidth=0)
    ax_a.bar(x + offset, first, width, color=METRIC_COLORS[name], linewidth=0,
             label="F1" if name == "f1" else name.capitalize())
ax_a.set_xticks(x)
ax_a.set_xticklabels([SHORT.get(g, g) for g in groups], rotation=20, ha="right")
ax_a.set_ylabel("Share of variance")
ax_a.set_ylim(0, 1)
ax_a.legend(loc="upper right", frameon=False)
ax_a.spines[["top", "right"]].set_visible(False)
letter(ax_a, "A")

# (B) the additive model's main effects: what a one-at-a-time profile of each
# parameter looks like once averaged over every position of the other three.
model = models["f1"]
for a, factor in enumerate(FACTORS):
    profile = model.partial_dependence(a)
    ax_b.plot(np.linspace(0, 1, len(profile)), profile, color=FACTOR_COLORS[factor],
              linewidth=1.3, label=SHORT[factor])
ax_b.axhline(model.grand, color=MIDGREY, linewidth=0.7, linestyle=(0, (1, 2)))
ax_b.set_xlabel("Position within the swept range")
ax_b.set_ylabel("Marginal F1")
ax_b.set_xlim(0, 1)
ax_b.legend(loc="lower left", frameon=False)
ax_b.spines[["top", "right"]].set_visible(False)
letter(ax_b, "B")

# (C) the whole grid in the plane it is actually judged in.
ax_c.scatter(grid["recall"], grid["precision"], s=1.5, color=GREY, linewidth=0, rasterized=True)
ax_c.plot(front["recall"], front["precision"], color="#000000", linewidth=1.0, marker="o",
          markersize=2.0, label=f"Pareto front (n = {len(front)})")
ax_c.scatter([adopted["recall"]], [adopted["precision"]], s=34, marker="*",
             color=METRIC_COLORS["f1"], zorder=5, label="Adopted setting")
ax_c.scatter([best_f1["recall"]], [best_f1["precision"]], s=16, marker="D",
             facecolor="none", edgecolor=METRIC_COLORS["recall"], linewidth=0.9,
             zorder=5, label="Maximum F1")
ax_c.set_xlabel("Recall")
ax_c.set_ylabel("Precision")
ax_c.legend(loc="lower left", frameon=False)
ax_c.spines[["top", "right"]].set_visible(False)
letter(ax_c, "C")

# (D) the front that can actually inform a choice: the crediting rule held at
# the adopted value, so only the three set-defining parameters move.
floors = np.unique(at_half["size floor"])
ramp = plt.cm.viridis(np.linspace(0.08, 0.92, len(floors)))
for value, color in zip(floors, ramp):
    subset = at_half[at_half["size floor"] == value]
    ax_d.scatter(subset["recall"], subset["precision"], s=2.0, color=color,
                 linewidth=0, rasterized=True,
                 label=f"{value / 1000:g} kb" if value else "0")
ax_d.plot(front_half["recall"], front_half["precision"], color="#000000", linewidth=1.0,
          marker="o", markersize=2.0, label=f"Front (n = {len(front_half)})")
ax_d.scatter([adopted["recall"]], [adopted["precision"]], s=34, marker="*",
             color=METRIC_COLORS["f1"], zorder=5, label="Adopted setting")
ax_d.set_xlabel("Recall")
ax_d.set_ylabel("Precision")
ax_d.legend(loc="lower left", frameon=False, ncol=2, handletextpad=0.4, columnspacing=0.9,
            title="Size floor", title_fontsize=6.5, alignment="left")
ax_d.spines[["top", "right"]].set_visible(False)
letter(ax_d, "D")

fig.tight_layout(pad=0.7, w_pad=1.8, h_pad=1.4)
DEST.mkdir(parents=True, exist_ok=True)
fig.savefig(DEST / "sensitivity.png", dpi=600)
fig.savefig(DEST / "sensitivity.pdf")
print(f"wrote {DEST / 'sensitivity.png'}")
