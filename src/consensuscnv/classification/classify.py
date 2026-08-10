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

from consensuscnv.classification.intervals import IntervalSet
from consensuscnv.classification.pairs import CandidateSet, PairSelection, filter_candidates


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

def group_metrics(
    classification: Classification,
    by: str = "sample_idx"
) -> dict[int, ClassificationSummary]:
    """Break the metrics down by an interend column, keyed by its registry id."""

    query_group = getattr(classification.query, by)
    truth_group = getattr(classification.truth, by)
    width = int(max(query_group.max(initial=-1), truth_group.max(initial=-1)) + 1)

    n_query = np.bincount(query_group, minlength=width)
    n_truth = np.bincount(truth_group, minlength=width)
    n_tp = np.bincount(query_group, weights=classification.query_matched, minlength=width)
    n_found = np.bincount(truth_group, weights=classification.truth_matched, minlength=width)
    np_pairs = np.bincount(query_group, weights=classification.query_n_partners, minlength=width)

    return {
        group_id: ClassificationSummary(
            n_query=int(n_query[group_id]),
            n_truth=int(n_truth[group_id]),
            n_pairs=int(np_pairs[group_id]),
            n_true_positive=int(n_tp[group_id]),
            n_false_positive=int(n_query[group_id] - n_tp[group_id]),
            n_truth_found=int(n_found[group_id]),
            n_false_negative=int(n_truth[group_id] - n_found[group_id]),
            min_reciprocal_overlap=classification.min_reciprocal_overlap,
            max_padding=classification.max_padding,
        )
        for group_id in range(width)
        if n_query[group_id] > 0 or n_truth[group_id] > 0
    }


def _warn_if_truth_covers_extra_samples(candidates: CandidateSet) -> None:
    """Warn if the truth set contains samples not in the query set.

    This is a common source of confusion, because it can make the recall
    appear lower than expected. The user may have intended to filter the
    truth set to only include samples present in the query set.
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
