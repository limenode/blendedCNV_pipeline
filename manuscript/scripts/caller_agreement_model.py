"""Is caller agreement one population or two, and how does each respond to depth?

Where used
----------
Results -> "Sequence-based Consensus Call Set Construction":
    Table 2  size and composition of the agreement categories at 30x
    Table 3  observed against predicted counts, 30x
    Table 4  the two populations refit at every coverage
    Supplementary Table Agreement Categories: the Table 3 breakdown at 6x, 4x, 2x

Fits a single-population null model to the caller-agreement categories: all three
callers draw from one pool of N events and each detects a given event with its
own independent probability p. For the pair-only category that excludes caller
i, the ratio to the all-three category is p_i / (1 - p_i), so each rate follows
from one observed ratio and N follows from the all-three count.

The model has four parameters and the four multi-caller categories supply four
counts, so those categories are reproduced by construction and are not evidence
either way. The test is the three single-caller categories, predicted out of
sample. The model underestimates them by more than an order of magnitude at
every coverage, which is the basis for describing agreement as two populations
rather than one.

Confidence intervals are multinomial resamples of the seven category counts,
refitting the model on each draw, so they carry the sampling uncertainty of the
categories through to the fitted parameters.

    pixi run python manuscript/scripts/caller_agreement_model.py
"""

import glob
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from consensuscnv.callsets import collect_callsets, merge_components, read_bed_calls
from consensuscnv.callsets.registry import SOURCES, SVTYPES
from consensuscnv.classification.intervals import IntervalSet

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")
DEST = ROOT / "results" / "manuscript"
CALLERS = ("cnvpytor", "delly", "gatk")
LABELS = {"cnvpytor": "CNVpytor", "delly": "Delly", "gatk": "GATK-gCNV"}
COVERAGES = ("30x", "6x", "4x", "2x")
CONSENSUS_THRESHOLD = 0.5
N_BOOTSTRAP = 2000

# Every subset of the three callers except the empty one, as a bit pattern over
# CALLERS. Ordered most-agreement first, which is the order the tables use.
KEYS = sorted(("".join(k) for k in product("01", repeat=3) if "1" in k), key=lambda k: -k.count("1"))
rng = np.random.default_rng(0)


def category_label(key: str) -> str:
    return " + ".join(LABELS[c] for c, on in zip(CALLERS, key) if on == "1")


def merged_components(coverage: str) -> IntervalSet:
    """The 1/3 consensus call set for one coverage: every component, unfiltered."""
    beds = [
        bed
        for caller in CALLERS
        for bed in sorted(glob.glob(str(ROOT / "out" / f"{coverage}_Coverage" / caller / "*.bed")))
    ]
    return IntervalSet.from_merged(
        merge_components(
            collect_callsets(read_bed_calls(bed) for bed in beds),
            min_reciprocal_overlap=CONSENSUS_THRESHOLD,
        )
    )


def category_masks(merged: IntervalSet) -> dict[str, np.ndarray]:
    """Row mask per category. SOURCES ids follow first-appearance order, so each
    mask is rebuilt from its key rather than assumed to be 1/2/4."""
    bits = {caller: 1 << SOURCES.get(caller) for caller in CALLERS}
    assert not (merged.source_bits & ~sum(bits.values())).any(), "non-caller source present"
    return {
        key: merged.source_bits == sum(bits[c] for c, on in zip(CALLERS, key) if on == "1")
        for key in KEYS
    }


def fit(counts: np.ndarray) -> tuple[np.ndarray, float]:
    """Detection rates and pool size from the four multi-caller categories."""
    index = {key: i for i, key in enumerate(KEYS)}
    rates = []
    for position, _ in enumerate(CALLERS):
        without = "".join("0" if j == position else "1" for j in range(len(CALLERS)))
        ratio = counts[index["111"]] / max(counts[index[without]], 1)
        rates.append(ratio / (1 + ratio))
    rates = np.array(rates)
    return rates, counts[index["111"]] / np.prod(rates)


def predict(rates: np.ndarray, pool: float, key: str) -> float:
    return pool * np.prod([p if on == "1" else 1 - p for p, on in zip(rates, key)])


# --------------------------------------------------------------------------- #
# Per-coverage fits
# --------------------------------------------------------------------------- #
per_category: list[dict] = []
per_coverage: list[dict] = []
composition_rows: list[dict] = []

for coverage in COVERAGES:
    merged = merged_components(coverage)
    masks = category_masks(merged)
    counts = np.array([int(masks[key].sum()) for key in KEYS])
    rates, pool = fit(counts)

    # Multinomial resample of the categories, refit on each draw.
    draws = [fit(rng.multinomial(counts.sum(), counts / counts.sum())) for _ in range(N_BOOTSTRAP)]
    pool_ci = np.percentile([pool for _, pool in draws], [2.5, 97.5])

    private = int(sum(c for key, c in zip(KEYS, counts) if key.count("1") == 1))
    concordant = int(counts.sum()) - private
    per_coverage.append(
        {
            "coverage": coverage,
            "concordant": concordant,
            "private": private,
            "private_per_concordant": private / concordant,
            "pool": pool,
            "pool_lo": pool_ci[0],
            "pool_hi": pool_ci[1],
            **{f"p_{caller}": rate for caller, rate in zip(CALLERS, rates)},
        }
    )

    for key, count in zip(KEYS, counts):
        predicted = predict(rates, pool, key)
        per_category.append(
            {
                "coverage": coverage,
                "category": category_label(key),
                "n_callers": key.count("1"),
                "observed": int(count),
                "predicted": predicted,
                "excess": int(count) - predicted if key.count("1") == 1 else np.nan,
                "fitted": key.count("1") > 1,
            }
        )

    if coverage == "30x":
        # Size and composition, split by SVTYPE because the two directions do not
        # share a size regime: duplications are the larger population in every
        # category, so one median over both hides the difference being described.
        svtypes = np.array(SVTYPES.names)[merged.svtype_idx]

        def middle(values: np.ndarray) -> float:
            return float(np.median(values)) if values.size else float("nan")

        def spread(values: np.ndarray) -> float:
            """Median absolute deviation about the median, in base pairs."""
            return float(np.median(np.abs(values - np.median(values)))) if values.size else np.nan

        for key in KEYS:
            rows = masks[key]
            deletions = merged.lengths[rows & (svtypes == "DEL")]
            duplications = merged.lengths[rows & (svtypes == "DUP")]
            composition_rows.append(
                {
                    "category": category_label(key),
                    "components": int(rows.sum()),
                    "median_all": middle(merged.lengths[rows]),
                    "median_del": middle(deletions),
                    "median_dup": middle(duplications),
                    "mad_all": spread(merged.lengths[rows]),
                    "mad_del": spread(deletions),
                    "mad_dup": spread(duplications),
                    "pct_del": 100.0 * (svtypes[rows] == "DEL").mean(),
                    "pct_dup": 100.0 * (svtypes[rows] == "DUP").mean(),
                }
            )

categories = pd.DataFrame(per_category)
coverages = pd.DataFrame(per_coverage)
composition = pd.DataFrame(composition_rows)

DEST.mkdir(parents=True, exist_ok=True)
composition.to_csv(DEST / "caller_agreement_composition_30x.csv", index=False)
categories.to_csv(DEST / "caller_agreement_categories.csv", index=False)
coverages.to_csv(DEST / "caller_agreement_by_coverage.csv", index=False)

pd.set_option("display.width", 220)
print("=== Table 2: size and composition by category, 30x ===")
print(composition.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

print("\n=== Table 3: observed against predicted, 30x ===")
print(categories[categories["coverage"] == "30x"].drop(columns=["coverage", "n_callers"])
      .to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

print("\n=== Table 4: the two populations by coverage ===")
print(coverages.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

print("\n=== Supplementary: observed against predicted at the reduced coverages ===")
print(categories[categories["coverage"] != "30x"].drop(columns="n_callers")
      .to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

# How much of the concordant collapse is the shrinking pool, and how much is the
# falling detection rates? Hold the rates at their 30x values and refill the 2x
# pool: the gap between that and the observed count is the rate contribution.
concordant_fraction = sum(
    np.prod([p if on == "1" else 1 - p for p, on in zip(coverages.iloc[0][["p_cnvpytor", "p_delly", "p_gatk"]].to_numpy(), key)])
    for key in KEYS
    if key.count("1") > 1
)
low = coverages.iloc[-1]
print(
    f"\nconcordant collapse, decomposed: {coverages.iloc[0]['concordant']:,} at 30x -> "
    f"{low['pool'] * concordant_fraction:,.0f} from the smaller pool alone -> "
    f"{low['concordant']:,} once the rates fall too"
)
