"""Computation phase: consensus calling, benchmark merging, binary classification."""

from consensuscnv.computation.computation_driver import main as run_computation
from consensuscnv.computation.computation_functions import (
    run_binary_classification_script,
)
from consensuscnv.computation.consensus_calling import (
    compute_consensus_from_beds,
    merge_benchmarks,
)

__all__ = [
    "compute_consensus_from_beds",
    "merge_benchmarks",
    "run_binary_classification_script",
    "run_computation",
]
