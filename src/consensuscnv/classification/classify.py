"""Labelling query and truth rows from a CandidateSet
and calculating binary classification metrics.

Classification is two-sided:
    - every query row is a true positive if it matched at least one truth row;
    otherwise, it is a false positive.
    - every truth row is "found" if it matched at least one query row;
    otherwise, it is a false negative.

Matching is many-to-many, so `n_true_positive` and `n_truth_found`
may be different numbers."""

import warnings
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.ndimage import gaussian_filter1d

from consensuscnv.classification.intervals import IntervalSet
from consensuscnv.classification.pairs import CandidateSet, PairSelection, filter_candidates
from consensuscnv.utils import DistributionType


class ClassLabel(Enum):
    """The three row classes. TP and FP index query rows; FN indexes truth rows."""

    TRUE_POSITIVE = "TP"
    FALSE_POSITIVE = "FP"
    FALSE_NEGATIVE = "FN"

@dataclass(frozen=True, slots=True)
class ClassificationSummary:
    """Scalar counts and metrics for one parameter point."""

    n_query: int
    n_truth: int
    n_pairs: int
    n_true_positive: int
    n_false_positive: int
    n_truth_found: int
    n_false_negative: int
    min_reciprocal_overlap: float
    max_padding: int | None

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        total = self.n_true_positive + self.n_false_positive
        return self.n_true_positive / total if total else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        total = self.n_truth_found + self.n_false_negative
        return self.n_truth_found / total if total else 0.0

    def f_beta(self, beta: float = 1.0) -> float:
        """F-beta score = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)"""
        p, r = self.precision, self.recall
        beta_squared = beta**2
        denominator = beta_squared * p + r
        return (1 + beta_squared) * (p * r) / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        """F1 score = 2 * (precision * recall) / (precision + recall)"""
        return self.f_beta(beta=1.0)

@dataclass(frozen=True, slots=True)
class MatchTopology:
    """How the matched pairs distribute over rows, for one parameter point.

    `query_fan_out[k]` is the number of query rows with exactly k truth partners;
    `truth_fan_in[k]` the number of truth rows with exactly k query partners.

    Notes:
        - The two sides are not symmetric. A query row with many truth partners
          is one call spanning a run of benchmark fragments; a truth row with
          many query partners is one benchmark call the caller broke up.
        - Above a reciprocal-overlap threshold of ~0.5 the matching is forced
          1:1 whenever both sets are internally disjoint, so this is a
          diagnostic for the low-threshold and padding regimes.
    """

    query_fan_out: np.ndarray = field(repr=False)
    truth_fan_in: np.ndarray = field(repr=False)
    min_reciprocal_overlap: float = 0.0
    max_padding: int | None = None

    @property
    def n_pairs(self) -> int:
        return int((self.query_fan_out * np.arange(len(self.query_fan_out))).sum())

    @property
    def n_matched_query(self) -> int:
        """Query rows with at least one truth partner -- the true positives."""
        return int(self.query_fan_out[1:].sum())

    @property
    def n_matched_truth(self) -> int:
        """Truth rows with at least one query partner -- the found calls."""
        return int(self.truth_fan_in[1:].sum())

    @property
    def n_query_multi(self) -> int:
        """Query rows spanning more than one truth call."""
        return int(self.query_fan_out[2:].sum())

    @property
    def n_truth_multi(self) -> int:
        """Truth calls broken across more than one query row."""
        return int(self.truth_fan_in[2:].sum())

    @property
    def max_query_partners(self) -> int:
        return len(self.query_fan_out) - 1

    @property
    def max_truth_partners(self) -> int:
        return len(self.truth_fan_in) - 1

    @property
    def mean_query_fan_out(self) -> float:
        """Truth partners per matched query row. 1.0 when the matching is 1:1."""
        n = self.n_matched_query
        return self.n_pairs / n if n else 0.0

    @property
    def is_one_to_one(self) -> bool:
        return self.n_query_multi == 0 and self.n_truth_multi == 0

@dataclass(frozen=True, slots=True)
class Classification:
    """
    Per-row labels for one parameter point, plus the partner counts behind them.

    Notes:
        - `query_matched` and `truth_matched` are boolean masks for
        `candidates.query` and `candidates.truth` respectively.
        - The partner counts are kept to see the many-to-many structure.
    """

    query_matched: np.ndarray
    truth_matched: np.ndarray
    query_n_partners: np.ndarray
    truth_n_partners: np.ndarray

    min_reciprocal_overlap: float
    max_padding: int | None
    candidates: CandidateSet = field(repr=False)

    @property
    def query(self) -> IntervalSet:
        return self.candidates.query

    @property
    def truth(self) -> IntervalSet:
        return self.candidates.truth

    @property
    def n_true_positive(self) -> int:
        return int(self.query_matched.sum())

    @property
    def n_false_positive(self) -> int:
        return len(self.query_matched) - self.n_true_positive

    @property
    def n_truth_found(self) -> int:
        return int(self.truth_matched.sum())

    @property
    def n_false_negative(self) -> int:
        return len(self.truth_matched) - self.n_truth_found

    @property
    def true_positive_rows(self) -> np.ndarray:
        """Query row indices labelled TP."""
        return np.flatnonzero(self.query_matched)

    @property
    def false_positive_rows(self) -> np.ndarray:
        """Query row indices labelled FP."""
        return np.flatnonzero(~self.query_matched)

    @property
    def false_negative_rows(self) -> np.ndarray:
        """Truth row indices labelled FN."""
        return np.flatnonzero(~self.truth_matched)

    def rows_for(self, label: ClassLabel) -> np.ndarray:
        """Return the row indices for a given class label."""
        if label == ClassLabel.TRUE_POSITIVE:
            return self.true_positive_rows
        elif label == ClassLabel.FALSE_POSITIVE:
            return self.false_positive_rows
        elif label == ClassLabel.FALSE_NEGATIVE:
            return self.false_negative_rows

    def summary(self) -> ClassificationSummary:
        """Collapse to scalars."""
        n_query = len(self.query_matched)
        n_truth = len(self.truth_matched)
        n_true_positive = self.n_true_positive
        n_truth_found = self.n_truth_found
        return ClassificationSummary(
            n_query=n_query,
            n_truth=n_truth,
            n_pairs=int(self.query_n_partners.sum()),
            n_true_positive=n_true_positive,
            n_false_positive=n_query - n_true_positive,
            n_truth_found=n_truth_found,
            n_false_negative=n_truth - n_truth_found,
            min_reciprocal_overlap=self.min_reciprocal_overlap,
            max_padding=self.max_padding,
        )

def classify(
    candidates: CandidateSet,
    selection: PairSelection | None = None,
    *,
    min_reciprocal_overlap: float = 0.0,
    max_padding: int | None = None,
    allow_mixed: bool = True,
    validate: bool = True
):
    """Label every query row as TP/FP and every truth row as found/FN.

    Pass a prebuilt `selection` to reuse one produced by `filter_candidates`;
    otherwise it is derived from the threholds.
    """
    if selection is None:
        selection = filter_candidates(
            candidates,
            min_reciprocal_overlap=min_reciprocal_overlap,
            max_padding=max_padding,
            allow_mixed=allow_mixed,
        )

    if validate:
        _warn_if_truth_covers_extra_samples(candidates)

    query_n_partners = np.bincount(selection.query_row, minlength=len(candidates.query))
    truth_n_partners = np.bincount(selection.truth_row, minlength=len(candidates.truth))

    return Classification(
        query_matched=query_n_partners > 0,
        truth_matched=truth_n_partners > 0,
        query_n_partners=query_n_partners,
        truth_n_partners=truth_n_partners,
        min_reciprocal_overlap=selection.min_reciprocal_overlap,
        max_padding=selection.max_padding,
        candidates=candidates,
    )

def match_topology(classification: Classification) -> MatchTopology:
    """Fan-out distributions for one parameter point.

    One `bincount` per side. Reads the parameters off the Classification, so a
    MatchTopology is always labelled with the point it was measured at.
    """
    return MatchTopology(
        query_fan_out=np.bincount(classification.query_n_partners),
        truth_fan_in=np.bincount(classification.truth_n_partners),
        min_reciprocal_overlap=classification.min_reciprocal_overlap,
        max_padding=classification.max_padding,
    )

def _counts_by_group(
    classification: Classification,
    query_group: np.ndarray,
    truth_group: np.ndarray,
    width: int,
) -> tuple[np.ndarray, ...]:
    """Per-group (n_query, n_truth, n_true_positive, n_truth_found, n_pairs).

    One scatter-add per column. Query-side columns are keyed by `query_group`
    and truth-side ones by `truth_group`, which is what makes the two sides'
    denominators independent.
    """
    return (
        np.bincount(query_group, minlength=width),
        np.bincount(truth_group, minlength=width),
        np.bincount(query_group, weights=classification.query_matched,
                    minlength=width).astype(np.int64),
        np.bincount(truth_group, weights=classification.truth_matched,
                    minlength=width).astype(np.int64),
        np.bincount(query_group, weights=classification.query_n_partners,
                    minlength=width).astype(np.int64),
    )

def _summary_from_counts(
    counts: tuple[np.ndarray, ...], index: int, classification: Classification
) -> ClassificationSummary:
    """Pull one group out of `_counts_by_group` output as a ClassificationSummary."""
    n_query, n_truth, n_tp, n_found, n_pairs = counts
    return ClassificationSummary(
        n_query=int(n_query[index]),
        n_truth=int(n_truth[index]),
        n_pairs=int(n_pairs[index]),
        n_true_positive=int(n_tp[index]),
        n_false_positive=int(n_query[index] - n_tp[index]),
        n_truth_found=int(n_found[index]),
        n_false_negative=int(n_truth[index] - n_found[index]),
        min_reciprocal_overlap=classification.min_reciprocal_overlap,
        max_padding=classification.max_padding,
    )

def group_metrics(
    classification: Classification,
    by: str = "sample_idx"
) -> dict[int, ClassificationSummary]:
    """Break the metrics down by an interend column, keyed by its registry id."""

    query_group = getattr(classification.query, by)
    truth_group = getattr(classification.truth, by)
    width = int(max(query_group.max(initial=-1), truth_group.max(initial=-1)) + 1)

    counts = _counts_by_group(classification, query_group, truth_group, width)
    n_query, n_truth = counts[0], counts[1]

    return {
        group_id: _summary_from_counts(counts, group_id, classification)
        for group_id in range(width)
        if n_query[group_id] > 0 or n_truth[group_id] > 0
    }


@dataclass(frozen=True, slots=True)
class SizeBinning:
    """Row -> size-bin assignment for one query/truth pair.

    Depends only on the interval sizes, never on a classification threshold, so
    one SizeBinning serves every parameter point -- the same hoist that makes
    `CandidateSet` reusable across a sweep.

    Bin ``k`` covers ``[lower_edges[k], lower_edges[k + 1])``: bin 0 is
    everything below ``edges[0]`` and bin ``len(edges)`` everything at or above
    ``edges[-1]``.
    """

    edges: np.ndarray
    query_bin: np.ndarray = field(repr=False)
    truth_bin: np.ndarray = field(repr=False)

    def __len__(self) -> int:
        return len(self.edges) + 1

    @property
    def lower_edges(self) -> np.ndarray:
        """Left edge of each bin -- the natural x values for a plot."""
        return np.concatenate(([0], self.edges))

    @classmethod
    def from_candidates(cls, candidates: CandidateSet, edges: np.ndarray) -> "SizeBinning":
        edges = np.asarray(edges)
        return cls(
            edges=edges,
            query_bin=np.searchsorted(edges, candidates.query.lengths, side="right"),
            truth_bin=np.searchsorted(edges, candidates.truth.lengths, side="right"),
        )

    @classmethod
    def at_every_size(cls, candidates: CandidateSet) -> "SizeBinning":
        """Bin at every distinct size present in either set.

        The metrics are step functions that can only change where a row's size
        actually falls, so this resolution is exact -- it reproduces a
        sort-and-step sweep point for point. Distinct sizes are far fewer than
        rows, and `bincount` is flat in the number of bins, so it is also cheap.
        """
        return cls.from_candidates(
            candidates,
            np.union1d(np.unique(candidates.query.lengths), np.unique(candidates.truth.lengths)),
        )

@dataclass(frozen=True, slots=True)
class SizeMetrics:
    """Classification metrics as a function of CNV size.

    Columnar rather than a list of summaries, because these are meant to be
    plotted. `precision` / `recall` / `f1` are NaN where the corresponding
    denominator is empty, so a plot breaks the line instead of drawing a drop to
    zero that would read as a real collapse. Index into it for a
    `ClassificationSummary` of one bin, which keeps the scalar 0.0 convention.
    """

    lower_edges: np.ndarray
    n_query: np.ndarray
    n_truth: np.ndarray
    n_pairs: np.ndarray
    n_true_positive: np.ndarray
    n_truth_found: np.ndarray

    distribution: DistributionType
    min_reciprocal_overlap: float
    max_padding: int | None

    def __len__(self) -> int:
        return len(self.lower_edges)

    def __getitem__(self, index: int) -> ClassificationSummary:
        return ClassificationSummary(
            n_query=int(self.n_query[index]),
            n_truth=int(self.n_truth[index]),
            n_pairs=int(self.n_pairs[index]),
            n_true_positive=int(self.n_true_positive[index]),
            n_false_positive=int(self.n_query[index] - self.n_true_positive[index]),
            n_truth_found=int(self.n_truth_found[index]),
            n_false_negative=int(self.n_truth[index] - self.n_truth_found[index]),
            min_reciprocal_overlap=self.min_reciprocal_overlap,
            max_padding=self.max_padding,
        )

    @property
    def n_false_positive(self) -> np.ndarray:
        return self.n_query - self.n_true_positive

    @property
    def n_false_negative(self) -> np.ndarray:
        return self.n_truth - self.n_truth_found

    @property
    def precision(self) -> np.ndarray:
        """Share of query calls of this size that matched something."""
        return _safe_ratio(self.n_true_positive, self.n_query)

    @property
    def recall(self) -> np.ndarray:
        """Share of truth calls of this size that were found."""
        return _safe_ratio(self.n_truth_found, self.n_truth)

    def f_beta(self, beta: float = 1.0) -> np.ndarray:
        p, r = self.precision, self.recall
        beta_squared = beta**2
        return _safe_ratio((1 + beta_squared) * p * r, beta_squared * p + r)

    @property
    def f1(self) -> np.ndarray:
        return self.f_beta(beta=1.0)

def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise division, NaN where the denominator is zero or undefined."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    valid = denominator > 0
    out = np.full(denominator.shape, np.nan)
    np.divide(numerator, denominator, out=out, where=valid)
    return out

def size_metrics(
    classification: Classification,
    bins: "SizeBinning | np.ndarray",
    *,
    distribution: DistributionType = DistributionType.DENSITY,
) -> SizeMetrics:
    """Classification metrics stratified by CNV size.

    Query rows are binned by query size and truth rows by truth size, so
    `precision` at a size is over calls of that size and `recall` is over truth
    events of that size. They are different populations by construction -- there
    is no single "number of CNVs of size s" that both share.

    `distribution` selects how bins accumulate:
        DENSITY                  -- calls whose size falls in the bin
        CUMULATIVE               -- calls at or below the bin's upper edge
        COMPLEMENTARY_CUMULATIVE -- calls at or above the bin's lower edge

    Both cumulative forms are exact at their edges: they are the same running
    totals a sort-and-step sweep produces, evaluated at the chosen sizes rather
    than at every distinct one. Use `SizeBinning.at_every_size` for the full
    curve. Pass a prebuilt `SizeBinning` to reuse the assignment across a sweep;
    an array of edges is binned on the spot.
    """
    if not isinstance(bins, SizeBinning):
        bins = SizeBinning.from_candidates(classification.candidates, bins)

    counts = _counts_by_group(classification, bins.query_bin, bins.truth_bin, len(bins))

    if distribution is DistributionType.CUMULATIVE:
        counts = tuple(np.cumsum(column) for column in counts)
    elif distribution is DistributionType.COMPLEMENTARY_CUMULATIVE:
        counts = tuple(np.cumsum(column[::-1])[::-1] for column in counts)

    n_query, n_truth, n_tp, n_found, n_pairs = counts
    return SizeMetrics(
        lower_edges=bins.lower_edges,
        n_query=n_query,
        n_truth=n_truth,
        n_pairs=n_pairs,
        n_true_positive=n_tp,
        n_truth_found=n_found,
        distribution=distribution,
        min_reciprocal_overlap=classification.min_reciprocal_overlap,
        max_padding=classification.max_padding,
    )

@dataclass(frozen=True, slots=True)
class SizeDensityCurve:
    """Kernel-smoothed metrics against size, plus the size densities behind them.

    `precision` here is a ratio of two kernel density estimates sharing one
    bandwidth: the density of matched query calls over the density of all query
    calls, at each size. That is the smooth counterpart of a DENSITY-binned
    `SizeMetrics` -- soft kernel weights in place of hard bin edges -- and it is
    a local rate, not a probability density; only `query_density` and
    `truth_density` integrate to one.

    `query_weight` / `truth_weight` are the effective number of calls behind
    each grid point (the summed kernel weights). Metrics are NaN wherever that
    falls below the requested floor, since a ratio estimated from two calls is
    noise however smooth it looks.
    """

    sizes: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    query_density: np.ndarray
    truth_density: np.ndarray
    query_weight: np.ndarray = field(repr=False)
    truth_weight: np.ndarray = field(repr=False)

    bandwidth: float
    min_effective_count: float
    min_reciprocal_overlap: float
    max_padding: int | None

    def __len__(self) -> int:
        return len(self.sizes)

    def f_beta(self, beta: float = 1.0) -> np.ndarray:
        p, r = self.precision, self.recall
        beta_squared = beta**2
        return _safe_ratio((1 + beta_squared) * p * r, beta_squared * p + r)

    @property
    def f1(self) -> np.ndarray:
        return self.f_beta(beta=1.0)

def size_density_curve(
    classification: Classification,
    *,
    bandwidth: float = 0.12,
    n_points: int = 512,
    size_range: tuple[float, float] | None = None,
    min_effective_count: float = 20.0,
) -> SizeDensityCurve:
    """Kernel-smoothed precision and recall against CNV size.

    Sizes are smoothed in log10 space, where CNV size distributions are roughly
    unimodal and a single `bandwidth` (in decades -- 0.12 is about +/-32%) is
    meaningful across four orders of magnitude. A fixed-width kernel in raw base
    pairs would be far too wide at 1 kb and far too narrow at 1 Mb.

    Implemented as a fine histogram convolved with a Gaussian rather than an
    O(n_points * n_rows) kernel sum, which is what makes it affordable over large
    numbers of truth rows. Numerator and denominator are smoothed identically, so
    the boundary bias of the two densities largely cancels in their ratio.
    """
    query_size = classification.query.lengths
    truth_size = classification.truth.lengths

    # log10 of a 0-length interval is undefined; clamp to 1 bp.
    log_query = np.log10(np.maximum(query_size, 1))
    log_truth = np.log10(np.maximum(truth_size, 1))

    if size_range is None:
        low = float(min(log_query.min(), log_truth.min()))
        high = float(max(log_query.max(), log_truth.max()))
    else:
        low, high = (float(np.log10(max(bound, 1))) for bound in size_range)
    if not high > low:
        high = low + 1.0

    grid = np.linspace(low, high, n_points)
    step = (high - low) / (n_points - 1)
    sigma = bandwidth / step

    def smooth(log_sizes: np.ndarray, matched: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        index = np.clip(np.rint((log_sizes - low) / step).astype(np.int64), 0, n_points - 1)
        total = np.bincount(index, minlength=n_points).astype(np.float64)
        hits = np.bincount(index, weights=matched, minlength=n_points)
        return (
            gaussian_filter1d(total, sigma, mode="constant"),
            gaussian_filter1d(hits, sigma, mode="constant"),
        )

    query_total, query_hits = smooth(log_query, classification.query_matched)
    truth_total, truth_hits = smooth(log_truth, classification.truth_matched)

    # gaussian_filter1d normalises its kernel to sum 1, so multiplying back by
    # sqrt(2*pi)*sigma recovers the summed unnormalised weights -- the effective
    # number of calls each grid point is estimated from.
    weight_scale = np.sqrt(2.0 * np.pi) * sigma
    query_weight = query_total * weight_scale
    truth_weight = truth_total * weight_scale

    precision = _safe_ratio(query_hits, np.where(query_weight >= min_effective_count,
                                                 query_total, 0.0))
    recall = _safe_ratio(truth_hits, np.where(truth_weight >= min_effective_count,
                                              truth_total, 0.0))

    return SizeDensityCurve(
        sizes=np.power(10.0, grid),
        precision=precision,
        recall=recall,
        query_density=query_total / max(query_total.sum() * step, 1e-300),
        truth_density=truth_total / max(truth_total.sum() * step, 1e-300),
        query_weight=query_weight,
        truth_weight=truth_weight,
        bandwidth=bandwidth,
        min_effective_count=min_effective_count,
        min_reciprocal_overlap=classification.min_reciprocal_overlap,
        max_padding=classification.max_padding,
    )

def _warn_if_truth_covers_extra_samples(candidates: CandidateSet) -> None:
    """Warn if the truth set contains samples not in the query set.

    If warning shows, recall may be lower than expected due a higher truth call count.
    """
    query_samples = np.unique(candidates.query.sample_idx)
    truth_samples = np.unique(candidates.truth.sample_idx)
    extra = np.setdiff1d(truth_samples, query_samples, assume_unique=True)
    if extra.size > 0:
        warnings.warn(
            f"Truth set contains {len(extra)} sample(s) not present in query set: "
            f"({extra.size + query_samples.size} truth samples vs. {query_samples.size} "
            f"query samples). "
            f"Extra samples: {extra}. This may lead to lower recall than expected. ",
            stacklevel=2
        )
