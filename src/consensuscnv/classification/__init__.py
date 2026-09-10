"""Relating intervals across two call sets: matching, metrics, and labels.

The mirror of `consensuscnv.callsets`, which relates intervals within one set.
Both are threshold-independent: the build records every pair that could ever be
joined, and filtering is a `searchsorted` plus a slice, so one built object serves
every parameter point.

    intervals.py  IntervalSet -- one side of a comparison
    pairs.py      CandidateSet, build_candidates, filter_candidates
    classify.py   classify, Classification, size_density_curve
    labels.py     write_intervals_bed, write_labels -- the TP / FP / FN rows
"""
