"""Benchmark composition by variant class, overall and against size.

Where used
----------
Results -> "Input Call Sets After Parsing":
    Table 1     the benchmark rows
    Figure 2    merged truth set against size: interval counts per class (A) and
                the duplication share, merged and per source (B)
    the benchmark composition paragraph and the numbers it quotes

Reports, per benchmark source and for the three sources merged into one truth
set: interval count, DEL/DUP split, median size, and the fraction of intervals
reaching 1 kb and 10 kb. The figure draws the same population against size,
because the class composition of the truth set is not a constant: duplications
are a small minority below the 1 kb floor and the majority above 10 kb, and that
profile is the denominator behind every class-split recall curve downstream.

Merging uses the same settings as the truth side of the classification
(`max_padding=0`), so the merged row is the population the classifier actually
sees, not a naive concatenation. The per-source rows and curves merge each
source on its own with the same setting.

    pixi run python manuscript/scripts/benchmark_composition.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from consensuscnv.callsets import collect_callsets, merge_components, read_bed_calls
from consensuscnv.callsets.registry import SVTYPES, seed_chromosomes
from consensuscnv.classification.intervals import IntervalSet
from consensuscnv.utils import read_genome_file

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")

# Chromosome ids have to order the genome rather than the order the BEDs
# happen to be read in, so the registry is seeded before any CallSet is built.
seed_chromosomes(read_genome_file(ROOT / "data" / "genome_primary_hg38.txt"))
BENCHMARKS = ("1000G", "HGSVC3", "ont_vienna")
LABELS = {"1000G": "1000G high-coverage", "HGSVC3": "HGSVC3", "ont_vienna": "ONT Vienna"}
DEST = ROOT / "results" / "manuscript"
FIGURES = ROOT / "results" / "benchmark_composition"

FLOORS = (1_000, 10_000)
SIZE_FLOOR = 1_000

# Log-spaced size bins for the figure: five per decade from 50 bp to ~1.3 Mb.
# 50 bp is the minimum size of a structural variant in the 1000G and HGSVC3
# releases; the 3,150 smaller intervals in the merged set are all SVAN tandem
# duplications from ONT Vienna, and drawing them would show a 100% duplication
# share that is a definition rather than a property of the genome. A share is
# drawn only where a bin holds at least MIN_BIN_COUNT intervals.
BIN_EDGES = np.power(10.0, np.arange(np.log10(50), 6.3, 0.2))
BIN_CENTRES = np.sqrt(BIN_EDGES[:-1] * BIN_EDGES[1:])
MIN_BIN_COUNT = 30


def describe(label: str, sizes: np.ndarray, svtypes: np.ndarray) -> dict:
    row = {
        "call_set": label,
        "n_intervals": sizes.size,
        "n_del": int((svtypes == "DEL").sum()),
        "n_dup": int((svtypes == "DUP").sum()),
        "pct_del": 100.0 * (svtypes == "DEL").mean(),
        "median_size": float(np.median(sizes)),
    }
    for floor in FLOORS:
        row[f"pct_ge_{floor}"] = 100.0 * (sizes >= floor).mean()
        row[f"n_dup_ge_{floor}"] = int(((svtypes == "DUP") & (sizes >= floor)).sum())
    return row


def as_interval_set(paths) -> IntervalSet:
    calls = collect_callsets(read_bed_calls(p) for p in paths)
    return IntervalSet.from_merged(merge_components(calls, max_padding=0))


def binned(sizes: np.ndarray, is_dup: np.ndarray) -> pd.DataFrame:
    """Interval counts per class and the duplication share, per log-size bin."""
    n_all, _ = np.histogram(sizes, bins=BIN_EDGES)
    n_dup, _ = np.histogram(sizes[is_dup], bins=BIN_EDGES)
    share = np.where(n_all >= MIN_BIN_COUNT, n_dup / np.maximum(n_all, 1), np.nan)
    return pd.DataFrame({
        "size": BIN_CENTRES, "n_intervals": n_all, "n_del": n_all - n_dup,
        "n_dup": n_dup, "pct_dup": 100.0 * share,
    })


rows = []
curves = {}
all_paths = []
for name in BENCHMARKS:
    paths = sorted((ROOT / "out" / "benchmark" / name).glob("*.bed"))
    all_paths.extend(paths)
    intervals = as_interval_set(paths)
    svtypes = np.array(SVTYPES.names)[intervals.svtype_idx]
    rows.append(describe(name, intervals.lengths, svtypes))
    curves[name] = binned(intervals.lengths, svtypes == "DUP")

merged = as_interval_set(all_paths)
merged_svtypes = np.array(SVTYPES.names)[merged.svtype_idx]
rows.append(describe("merged truth set", merged.lengths, merged_svtypes))
curves["merged truth set"] = binned(merged.lengths, merged_svtypes == "DUP")

table = pd.DataFrame(rows)
DEST.mkdir(parents=True, exist_ok=True)
table.to_csv(DEST / "benchmark_composition.csv", index=False)
by_size = pd.concat(
    [frame.assign(call_set=name) for name, frame in curves.items()], ignore_index=True
)[["call_set", "size", "n_intervals", "n_del", "n_dup", "pct_dup"]]
by_size.to_csv(DEST / "benchmark_composition_by_size.csv", index=False)


# --------------------------------------------------------------------------- #
# Figure 2 -- the merged truth set against size
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

# Deletions in dark grey, duplications in the one Okabe-Ito hue no other figure
# uses; the size floor as a thin rule in both panels.
CLASS_COLORS = {"DEL": "#4D4D4D", "DUP": "#CC79A7"}
SOURCE_STYLES = {
    "merged truth set": {"color": "#000000", "linestyle": "-", "linewidth": 1.6},
    "1000G": {"color": "#000000", "linestyle": (0, (4, 1.6)), "linewidth": 1.0},
    "ont_vienna": {"color": "#000000", "linestyle": (0, (1, 1.4)), "linewidth": 1.2},
}
XTICKS = ([1e2, 1e3, 1e4, 1e5, 1e6], ["100 bp", "1 kb", "10 kb", "100 kb", "1 Mb"])


def panel(ax, letter: str, x: float = -0.16) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.4)
    ax.text(x, 1.03, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.45))

# (A) Intervals per bin by class. Log y, because the truth set is two orders of
# magnitude denser below the floor than in the range the callers operate in,
# and the crossing of the two curves is the point of the panel.
ax = axes[0]
frame = curves["merged truth set"]
for label, column in (("DEL", "n_del"), ("DUP", "n_dup")):
    ax.plot(frame["size"], frame[column].where(frame[column] > 0), color=CLASS_COLORS[label],
            linewidth=1.4, zorder=3)
ax.axvline(SIZE_FLOOR, color="#9E9E9E", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(50, 2e6)
ax.set_xticks(*XTICKS)
ax.set_xlabel("Interval size", labelpad=2)
ax.set_ylabel("Benchmark intervals per bin", labelpad=2)
ax.legend(handles=[Line2D([], [], color=CLASS_COLORS[k], linewidth=1.4, label=v)
                   for k, v in (("DEL", "Deletions"), ("DUP", "Duplications"))],
          frameon=False, loc="upper right", handlelength=1.6, borderpad=0,
          labelspacing=0.22, handletextpad=0.5)
panel(ax, "A")

# (B) Duplication share per bin, merged and per source. HGSVC3 carries no
# duplications and is left off rather than drawn as a zero line.
ax = axes[1]
for name in ("merged truth set", "1000G", "ont_vienna"):
    frame = curves[name]
    ax.plot(frame["size"], frame["pct_dup"], zorder=3, **SOURCE_STYLES[name])
ax.axvline(SIZE_FLOOR, color="#9E9E9E", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
ax.axhline(50, color="#9E9E9E", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
ax.set_xscale("log")
ax.set_xlim(50, 2e6)
ax.set_xticks(*XTICKS)
ax.set_ylim(0, 100)
ax.set_xlabel("Interval size", labelpad=2)
ax.set_ylabel("Duplications (%)", labelpad=2)
ax.legend(handles=[Line2D([], [], label={"merged truth set": "Merged truth set"}.get(n, LABELS.get(n, n)),
                          **SOURCE_STYLES[n]) for n in ("merged truth set", "1000G", "ont_vienna")],
          frameon=False, loc="upper left", handlelength=2.4, borderpad=0,
          labelspacing=0.22, handletextpad=0.5)
panel(ax, "B")

fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.165, wspace=0.3)
FIGURES.mkdir(parents=True, exist_ok=True)
for suffix, dpi in ((".png", 600), (".pdf", None)):
    fig.savefig(FIGURES / f"benchmark_composition{suffix}", dpi=dpi)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
pd.set_option("display.width", 220)
print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

print("\n=== merged truth set against size ===")
frame = curves["merged truth set"]
print(frame.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
above_half = frame[frame["pct_dup"] >= 50]
if not above_half.empty:
    first = above_half.iloc[0]
    print(f"\nduplications first exceed half of a bin at {first['size']:,.0f} bp "
          f"({first['pct_dup']:.1f}% of {first['n_intervals']:,} intervals)")
below_floor = frame[frame["size"] < SIZE_FLOOR]
print(f"below the floor the share runs {below_floor['pct_dup'].min():.1f}-{below_floor['pct_dup'].max():.1f}%")
n_small = int((merged.lengths < BIN_EDGES[0]).sum())
n_small_dup = int(((merged.lengths < BIN_EDGES[0]) & (merged_svtypes == "DUP")).sum())
print(f"intervals below {BIN_EDGES[0]:.0f} bp: {n_small:,}, of which {n_small_dup:,} DUP")
for name in ("1000G", "ont_vienna"):
    f = curves[name]
    drawn = f[f["pct_dup"].notna()]
    print(f"{LABELS[name]}: share drawn over {drawn['size'].min():,.0f}-{drawn['size'].max():,.0f} bp; "
          f"peak {f['pct_dup'].max():.1f}% at {f.loc[f['pct_dup'].idxmax(), 'size']:,.0f} bp")

for floor in (SIZE_FLOOR, 10_000, 100_000):
    above = merged.lengths >= floor
    n_dup = int(((merged_svtypes == "DUP") & above).sum())
    print(f"above {floor:>7,} bp: {above.sum():>7,} intervals, {n_dup:>6,} DUP ({100 * n_dup / above.sum():.1f}%)")
print(f"\nwrote {DEST / 'benchmark_composition.csv'}, {DEST / 'benchmark_composition_by_size.csv'}, "
      f"{FIGURES / 'benchmark_composition.png'}")
