from dataclasses import dataclass, field

import numpy as np

from consensuscnv.callsets.registry import SAMPLES, SVTYPES
from consensuscnv.classification.intervals import IntervalSet


@dataclass(frozen=True, slots=True)
class PairSelection:
    """The pairs that count as a match at one parameter point."""

    query_row: np.ndarray
    truth_row: np.ndarray
    min_reciprocal_overlap: float
    max_padding: int | None

    def __len__(self) -> int:
        return len(self.query_row)

@dataclass(frozen=True, slots=True)
class CandidateSet:
    """Every pair that could ever match, as two key-sorted edge lists.

    Mirrors `CallSet`'s ov_/gap_ arrays: build once, then `filter_candidates`
    is a binary search plus a slice at every parameter point.
    """

    # overlapping pairs, ascending reciprocal overlap
    ov_q: np.ndarray
    ov_t: np.ndarray
    ov_key: np.ndarray

    # non-overlapping pairs, ascending base-pair distance
    gap_q: np.ndarray
    gap_t: np.ndarray
    gap_key: np.ndarray

    search_radius: int
    query: IntervalSet = field(repr=False)
    truth: IntervalSet = field(repr=False)

    def __len__(self) -> int:
        return len(self.ov_q) + len(self.gap_q)

    @property
    def n_query(self) -> int:
        return len(self.query)

    @property
    def n_truth(self) -> int:
        return len(self.truth)

def partition_ids(interval_set: IntervalSet) -> np.ndarray:
    """Fold (chrom, svtype, sample) into one integer per row.

    Mixed radix. Because registries are process-global, the same formula
    applied to two different INtervalSets yields directly comparable ids.
    """
    n_svtypes = len(SVTYPES.names)
    n_samples = len(SAMPLES.names)
    return (
        (interval_set.chrom_idx.astype(np.int64) * n_svtypes + interval_set.svtype_idx)
        * n_samples
        + interval_set.sample_idx.astype(np.int64)
    )

def find_candidate_pairs(
    query: IntervalSet,
    truth: IntervalSet,
    *,
    search_radius: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Find every (query row, truth row) pair sharing a partition and lying within
    `search_radius` base pairs, as two index arrays.

    `search_radius=0` admits exactly-touching pairs and overlapping pairs.
    """

    if len(query) == 0 or len(truth) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    query_partition = partition_ids(query)
    truth_partition = partition_ids(truth)

    # 1. Give each partition a disjoint band. The band must be wider than any
    # coordinate a probe can reach, or a padded probe leaks into its neighbor.
    max_coord = int(max(query.ends.max(), truth.ends.max()))
    band_span = max_coord + (2 * search_radius) + 2
    _check_band_arithmetic(query_partition, truth_partition, band_span)

    query_band_base = query_partition * band_span
    truth_band_base = truth_partition * band_span
    truth_start_banded = truth_band_base + truth.starts
    truth_end_banded = truth_band_base + truth.ends

    # 2. Sort truth once by banded start, which orders by partition then coordinate.
    truth_order = np.argsort(truth_start_banded, kind="stable")
    sorted_truth_start = truth_start_banded[truth_order]
    sorted_truth_end = truth_end_banded[truth_order]

    # 3. Ends are not monotone, so they cannot be searched.
    # Use running maximum for upper bound calculation instead.
    running_max_end = np.maximum.accumulate(sorted_truth_end)

    # 4. Two probes per query row, clamped inside its own band (redundancy with 2 * search_radius).
    low_probe = np.maximum(
        query_band_base + query.starts - search_radius, query_band_base
    )
    high_probe = np.minimum(
        query_band_base + query.ends + search_radius, query_band_base + band_span - 1
    )

    window_lo = np.searchsorted(running_max_end, low_probe, side="left")
    window_hi = np.searchsorted(sorted_truth_start, high_probe, side="right")

    # 5. Expand the per-query windows into flat pair arrays.
    counts = np.maximum(window_hi - window_lo, 0)
    total = int(counts.sum())
    if total == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    query_row = np.repeat(np.arange(len(query), dtype=np.int64), counts)
    block_origin = np.repeat(np.cumsum(counts) - counts, counts)
    slot_in_block = np.arange(total, dtype=np.int64) - block_origin
    truth_row = truth_order[np.repeat(window_lo, counts) + slot_in_block]

    # 6. `window_lo` is conservative, so end test has to be re-applied exactly.
    keep = truth_end_banded[truth_row] >= low_probe[query_row]
    return query_row[keep], truth_row[keep]

def build_candidates(
    query: IntervalSet,
    truth: IntervalSet,
    *,
    search_radius: int = 0,
) -> CandidateSet:
    """Build a CandidateSet from a query and a truth IntervalSet.

    `search_radius` is the widest `filter_candidates(max_padding=...)` this
    CandidateSet will be able to serve. Leave it at 0 unless filtering by padding.
    """
    query_row, truth_row = find_candidate_pairs(query, truth, search_radius=search_radius)

    query_start, query_end = query.starts[query_row], query.ends[query_row]
    truth_start, truth_end = truth.starts[truth_row], truth.ends[truth_row]

    intersection = np.minimum(query_end, truth_end) - np.maximum(query_start, truth_start)
    overlaps = intersection > 0

    # A positive intersection implies both intervals are non-empty, so the
    # denominator is safe. Same definition as `CallSet.ov_key`: share of the
    # longer interval, which bounds both ratios at once.
    longest = np.maximum(query_end - query_start, truth_end - truth_start)
    ov_key = intersection[overlaps] / longest[overlaps]
    gap_key = -intersection[~overlaps]  # non-positive intersection is a distance

    ov_order = np.argsort(ov_key)
    gap_order = np.argsort(gap_key)

    return CandidateSet(
        ov_q=query_row[overlaps][ov_order],
        ov_t=truth_row[overlaps][ov_order],
        ov_key=ov_key[ov_order],
        gap_q=query_row[~overlaps][gap_order],
        gap_t=truth_row[~overlaps][gap_order],
        gap_key=gap_key[gap_order],
        search_radius=search_radius,
        query=query,
        truth=truth,
    )

def filter_candidates(
    candidates: CandidateSet,
    min_reciprocal_overlap: float = 0.0,
    max_padding: int | None = None,
    *,
    allow_mixed: bool = False,
) -> PairSelection:
    """Select the candidate pairs passing reciprocal overlap and/or padding thresholds.

    `max_padding=None` means no padding at all.
    `max_padding=0` admits exactly-touching pairs (gap distance 0).
    """
    if min_reciprocal_overlap > 0.0 and max_padding is not None and not allow_mixed:
        raise ValueError(
            f"min_reciprocal_overlap={min_reciprocal_overlap} and max_padding={max_padding} \
            are both set, but allow_mixed is False. This admits a pair sharing no bases at all \
            while rejecting one that overlaps below the threshold, which is not interpretable."
        )

    if max_padding is not None and max_padding > candidates.search_radius:
        raise ValueError(
            f"max_padding={max_padding} exceeds the search_radius this CandidateSet was built \
            with ({candidates.search_radius}). Pairs beyond that radius were never recorded, so \
            the result would be silently incomplete. Rebuild with a wider search_radius."
        )

    recip_idx = np.searchsorted(candidates.ov_key, min_reciprocal_overlap, side="left")

    if max_padding is None:
        return PairSelection(
            query_row=candidates.ov_q[recip_idx:],
            truth_row=candidates.ov_t[recip_idx:],
            min_reciprocal_overlap=min_reciprocal_overlap,
            max_padding=None,
        )

    padding_idx = np.searchsorted(candidates.gap_key, max_padding, side="right")

    if padding_idx == 0:
        query_row = candidates.ov_q[recip_idx:]
        truth_row = candidates.ov_t[recip_idx:]
    elif recip_idx == len(candidates.ov_key):
        query_row = candidates.gap_q[:padding_idx]
        truth_row = candidates.gap_t[:padding_idx]
    else:
        query_row = np.concatenate((candidates.ov_q[recip_idx:], candidates.gap_q[:padding_idx]))
        truth_row = np.concatenate((candidates.ov_t[recip_idx:], candidates.gap_t[:padding_idx]))

    return PairSelection(
        query_row=query_row,
        truth_row=truth_row,
        min_reciprocal_overlap=min_reciprocal_overlap,
        max_padding=max_padding,
    )



def _check_band_arithmetic(
    query_partition: np.ndarray,
    truth_partition: np.ndarray,
    band_span: int
) -> None:
    """Fail loudly if the banded coordinates would overflow int64"""

    highest_partition = int(max(query_partition.max(), truth_partition.max()))
    if highest_partition * band_span >= 2**62:
        raise ValueError(
            f"Band arithmetic would overflow int64: "
            f"highest_partition={highest_partition}, band_span={band_span}"
        )
