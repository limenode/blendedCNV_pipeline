"""Size distributions of every call set at the adopted parameters.

Where used
----------
Results -> "CNV Size Distribution Characteristics":
    Table 2   size statistics of the 30x call sets, the SNP array, and the benchmark
    Figure 9  size densities at 30x and of each consensus level across coverages
    every number quoted in that section

Also writes the two supplementary tables that section points at: the full
coverage x call set grid, and the same grid with the mean, quartiles, and
duplication share that the main table leaves out.

Nothing here touches the benchmark comparison. The section is descriptive: what
the callers produce and how that changes with depth, before any call is scored.
The adopted parameters still apply, because they decide which intervals exist --
the 1 kb floor on both sides and the 0.5 reciprocal overlap that builds a
consensus component. Padding and the classification threshold play no part in a
size distribution.

    pixi run python manuscript/scripts/size_distributions.py
"""

import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from consensuscnv.callsets import collect_callsets, merge_components, read_bed_calls
from consensuscnv.callsets.registry import SVTYPES
from consensuscnv.classification.intervals import IntervalSet

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")
DEST = ROOT / "results" / "size_distribution"
TABLES = ROOT / "results" / "manuscript"

COVERAGES = ("30x", "6x", "4x", "2x")
CALLERS = ("cnvpytor", "delly", "gatk")
LABELS = {"cnvpytor": "CNVpytor", "delly": "Delly", "gatk": "GATK-gCNV"}
BENCHMARKS = ("1000G", "HGSVC3", "ont_vienna")
CONSENSUS_LEVELS = (1, 2, 3)
SAMPLES = (
    "HG00096", "HG00171", "HG00268", "HG00513", "HG00731", "HG01596", "HG01890",
    "NA18989", "NA19129", "NA19238", "NA19331", "NA19347", "NA20847",
)

# Adopted parameters. The floor is applied to every set including the benchmark.
SIZE_FLOOR = 1_000
CONSENSUS_THRESHOLD = 0.5
BENCHMARK_PADDING = 0

# Same assignment as the parameterization figures: Okabe-Ito for the callers,
# a Purples ramp for the ordinal consensus levels, black for the benchmark.
# The SNP array is a second reference set rather than a query set, so it takes
# the mid grey that the other figures use for de-emphasis.
COLORS = {
    "CNVpytor": "#0072B2", "Delly": "#D55E00", "GATK-gCNV": "#009E73",
    "1/3": "#9E9AC8", "2/3": "#6A51A3", "3/3": "#3F007D",
    "SNP array": "#767676", "Benchmark": "#000000",
}
# Coverage is ordinal too, and has to stay distinguishable from every set colour
# above, so it takes a single-hue ramp of its own (ColorBrewer Blues, dark = deep).
COVERAGE_COLORS = {"30x": "#08519C", "6x": "#3182BD", "4x": "#6BAED6", "2x": "#BDD7E7"}

QUERY_ORDER = ["CNVpytor", "Delly", "GATK-gCNV", "1/3", "2/3", "3/3"]
REFERENCE_ORDER = ["SNP array", "Benchmark"]


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
def query_sets(coverage: str) -> dict[str, IntervalSet]:
    """The six call sets at one coverage: three callers raw, three consensus levels.

    Merged once at the default min_sources=1. Components are built from the edge
    selection alone and only afterwards dropped, so selecting n_sources >= k off
    the unfiltered result is exactly the set min_sources=k would have returned.

    Each caller is read from its own directory and passed through raw. Selecting
    a caller's bit out of the consensus instead would return the components that
    caller took part in, whose extents were set partly by the other two.
    """
    directory = ROOT / "out" / f"{coverage}_Coverage"
    consensus = IntervalSet.from_merged(
        merge_components(
            collect_callsets(read_bed_calls(bed) for bed in bed_paths(directory, CALLERS)),
            min_reciprocal_overlap=CONSENSUS_THRESHOLD,
        )
    )
    return {
        **{
            LABELS[caller]: floored(
                IntervalSet.from_callset(
                    collect_callsets(read_bed_calls(bed) for bed in bed_paths(directory, (caller,)))
                )
            )
            for caller in CALLERS
        },
        **{f"{k}/3": floored(consensus.select(consensus.n_sources >= k)) for k in CONSENSUS_LEVELS},
    }


benchmark = floored(
    IntervalSet.from_merged(
        merge_components(
            collect_callsets(read_bed_calls(bed) for bed in bed_paths(ROOT / "out" / "benchmark", BENCHMARKS)),
            max_padding=BENCHMARK_PADDING,
        )
    )
)
# The array was genotyped for the whole 1000 Genomes panel; only the thirteen
# samples this study sequenced belong in the comparison.
snp_array = floored(
    IntervalSet.from_callset(
        collect_callsets(read_bed_calls(str(ROOT / "out" / "SNP_Array" / "bed" / f"{s}.bed")) for s in SAMPLES)
    )
)

sets_by_coverage = {cov: query_sets(cov) for cov in COVERAGES}
references = {"SNP array": snp_array, "Benchmark": benchmark}

# The benchmark and the array are properties of the samples, not of the depth the
# genomes were sequenced to, so they must come out identical however they are reached.
assert len(np.unique(benchmark.sample_idx)) == len(SAMPLES)
assert len(np.unique(snp_array.sample_idx)) == len(SAMPLES)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def describe(name: str, coverage: str, intervals: IntervalSet) -> dict[str, object]:
    """Size statistics for one call set.

    Spread is reported as the median absolute deviation rather than the standard
    deviation: CNV sizes span three orders of magnitude with a heavy right tail,
    where a moment-based spread is set by a handful of the largest calls.
    """
    lengths = intervals.lengths.astype(np.float64)
    median = float(np.median(lengths))
    q1, q3 = np.percentile(lengths, [25, 75])
    return {
        "call set": name,
        "coverage": coverage,
        "n": len(lengths),
        "min": int(lengths.min()),
        "median": median,
        "mad": float(np.median(np.abs(lengths - median))),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


rows = [
    describe(name, cov, intervals)
    for cov in COVERAGES
    for name, intervals in sets_by_coverage[cov].items()
] + [describe(name, "--", intervals) for name, intervals in references.items()]
full = pd.DataFrame(rows)

# Duplication share. The registry id is resolved once, after every set is read,
# so the interning order cannot change underneath it.
dup_id = SVTYPES.intern("DUP")


def dup_share(intervals: IntervalSet) -> float:
    return float(np.mean(intervals.svtype_idx == dup_id))


full["dup share"] = [
    dup_share(sets_by_coverage[cov][name])
    for cov in COVERAGES
    for name in sets_by_coverage[cov]
] + [dup_share(references[name]) for name in references]

TABLES.mkdir(parents=True, exist_ok=True)
main = full[(full["coverage"].isin(("30x", "--")))][["call set", "coverage", "n", "min", "median", "mad", "max", "dup share"]]
main.to_csv(TABLES / "size_distribution_30x.csv", index=False)
full.to_csv(TABLES / "size_distribution_full.csv", index=False)

print(main.to_string(index=False))
print()
print(full.pivot_table(index="call set", columns="coverage", values="median").to_string())
print()
print(full.pivot_table(index="call set", columns="coverage", values="n").to_string())


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
})

GRID = np.linspace(np.log10(SIZE_FLOOR), 6.1, 512)


def density(intervals: IntervalSet) -> np.ndarray:
    """Kernel density of log10 size, evaluated on the shared grid.

    Estimated in log space because the distributions span three decades; a
    density on the raw scale is uninterpretable over that range.
    """
    return gaussian_kde(np.log10(intervals.lengths.astype(np.float64)))(GRID)


def draw(ax, curves: dict[str, np.ndarray], colors: dict[str, str], widths: dict[str, float] | None = None):
    for name, curve in curves.items():
        ax.plot(GRID, curve, color=colors[name], lw=(widths or {}).get(name, 1.0), label=name)
    ax.set_xlim(GRID[0], GRID[-1])
    ax.set_ylim(bottom=0)
    ax.set_xticks([3, 4, 5, 6])
    ax.set_xticklabels(["1 kb", "10 kb", "100 kb", "1 Mb"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.0))

# (A) every call set at 30x, against the two reference sets.
widths = {**{k: 1.0 for k in ("CNVpytor", "Delly", "GATK-gCNV")}, **{f"{k}/3": 1.5 for k in CONSENSUS_LEVELS}}
panel_a = {name: density(sets_by_coverage["30x"][name]) for name in QUERY_ORDER}
panel_a |= {name: density(references[name]) for name in REFERENCE_ORDER}
draw(axes[0, 0], panel_a, COLORS, {**widths, "Benchmark": 1.3, "SNP array": 1.3})
axes[0, 0].set_ylabel("Density")
axes[0, 0].legend(frameon=False, ncol=2, handlelength=1.4, columnspacing=1.0, loc="upper right")

# (B, C, D) each consensus level across the four coverages, with the benchmark
# repeated as a fixed reference. The three panels share a y-limit so the loss of
# density at the small end can be read across them rather than within each.
benchmark_curve = density(benchmark)
consensus_curves = {
    level: {cov: density(sets_by_coverage[cov][f"{level}/3"]) for cov in COVERAGES}
    for level in CONSENSUS_LEVELS
}
ymax = max(
    max(curve.max() for curves in consensus_curves.values() for curve in curves.values()),
    benchmark_curve.max(),
)
for ax, level in zip(axes.flat[1:], CONSENSUS_LEVELS):
    ax.plot(GRID, benchmark_curve, color="#000000", lw=0.9, ls=(0, (3, 2)), label="Benchmark")
    draw(ax, consensus_curves[level], COVERAGE_COLORS, {cov: 1.2 for cov in COVERAGES})
    ax.set_ylim(0, ymax * 1.05)
    ax.set_title(f"{level}-of-3 consensus", pad=3)
    if level == 1:
        ax.legend(frameon=False, handlelength=1.4, loc="upper right")

for ax in axes[1, :]:
    ax.set_xlabel("CNV size")
axes[1, 0].set_ylabel("Density")

for ax, letter in zip(axes.flat, "ABCD"):
    ax.text(-0.155, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

fig.tight_layout()
DEST.mkdir(parents=True, exist_ok=True)
fig.savefig(DEST / "size_distributions.pdf")
fig.savefig(DEST / "size_distributions.png", dpi=600)
plt.close(fig)


# --------------------------------------------------------------------------- #
# Supplementary figure: the same coverage axis for the individual callers
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.2), sharey=True)
for ax, caller in zip(axes, CALLERS):
    ax.plot(GRID, benchmark_curve, color="#000000", lw=0.9, ls=(0, (3, 2)), label="Benchmark")
    draw(ax, {cov: density(sets_by_coverage[cov][LABELS[caller]]) for cov in COVERAGES},
         COVERAGE_COLORS, {cov: 1.2 for cov in COVERAGES})
    ax.set_title(LABELS[caller], pad=3)
    ax.set_xlabel("CNV size")
axes[0].set_ylabel("Density")
axes[0].legend(frameon=False, handlelength=1.4, loc="upper right")
for ax, letter in zip(axes, "ABC"):
    ax.text(-0.10, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
fig.tight_layout()
fig.savefig(DEST / "size_distributions_per_caller.pdf")
fig.savefig(DEST / "size_distributions_per_caller.png", dpi=600)
plt.close(fig)

print(f"\nwrote {DEST}/size_distributions.{{pdf,png}} and _per_caller")
