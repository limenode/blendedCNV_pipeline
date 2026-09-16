"""Relating intervals across two call sets: matching, metrics, and labels.

The mirror of `consensuscnv.callsets`, which relates intervals within one set.
Both are threshold-independent: the build records every pair that could ever be
joined, and filtering is a `searchsorted` plus a slice, so one built object serves
every parameter point.

    intervals.py  IntervalSet -- one side of a comparison
    pairs.py      CandidateSet, build_candidates, filter_candidates
    classify.py   classify, Classification, size_density_curve
    labels.py     write_intervals_bed, write_labels -- the TP / FP / FN rows

    query = IntervalSet.from_merged(merged)          # or from_callset
    truth = IntervalSet.from_merged(truth_merged)
    candidates = build_candidates(query, truth)      # build once
    result = classify(candidates, min_reciprocal_overlap=0.5)

`build_candidates` takes the same `partition_by` as `build_callset`; with
``("svtype",)`` the pairs may join samples, which is how two groups of a cohort
are compared.
"""

from consensuscnv.classification.classify import (
    Classification,
    ClassificationSummary,
    ClassLabel,
    MatchTopology,
    SizeDensityCurve,
    classify,
    match_topology,
    size_density_curve,
)
from consensuscnv.classification.intervals import IntervalSet
from consensuscnv.classification.labels import write_intervals_bed, write_labels
from consensuscnv.classification.pairs import (
    CandidateSet,
    PairSelection,
    build_candidates,
    filter_candidates,
    partition_ids,
)

__all__ = [
    "CandidateSet",
    "ClassLabel",
    "Classification",
    "ClassificationSummary",
    "IntervalSet",
    "MatchTopology",
    "PairSelection",
    "SizeDensityCurve",
    "build_candidates",
    "classify",
    "filter_candidates",
    "match_topology",
    "partition_ids",
    "size_density_curve",
    "write_intervals_bed",
    "write_labels",
]
