"""Computation phase: consensus calling, benchmark merging, binary classification."""

from consensuscnv.computation.computation_driver import run_computation
from consensuscnv.computation.computation_functions import (
    ClassificationResult,
    classify_calls,
    run_binary_classification_script,
)
from consensuscnv.computation.consensus_calling import (
    compute_consensus_from_beds,
    merge_benchmarks,
)

__all__ = [
    "ClassificationResult",
    "classify_calls",
    "compute_consensus_from_beds",
    "merge_benchmarks",
    "run_binary_classification_script",
    "run_computation",
]
