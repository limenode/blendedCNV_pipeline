# %% Imports and constants

import glob
from pathlib import Path

from consensuscnv.callsets import (
    CallSet,
    MergedCallSet,
    collect_callsets,
    merge_components,
    read_bed_calls,
)

from consensuscnv.classification.intervals import IntervalSet

BENCHMARK_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark")
CONTROL_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/SNP_Array")
QUERY_DIRS = {
    "30x": Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/30x_Coverage"),
    "6x": Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/6x_Coverage"),
    "4x": Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/4x_Coverage"),
    "2x": Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/2x_Coverage"),
}
RESULTS_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/results")

# The detectable size domain.
SIZE_FLOOR = 1_000

def evaluation_intervals(merged: MergedCallSet) -> IntervalSet:
    """Merged calls as an IntervalSet, restricted to the detectable size domain."""
    return IntervalSet.from_merged(merged).filter_by_size(min_size=SIZE_FLOOR)

# %%
QUERY_DIR = QUERY_DIRS["30x"]
# 1. Benchmark
benchmark_callset = collect_callsets(read_bed_calls(bed) for bed in glob.glob(str(BENCHMARK_DIR / "*/*.bed")))
benchmark_merged = merge_components(benchmark_callset, max_padding=0)
benchmark_merged_interval_set = evaluation_intervals(benchmark_merged)
print("Benchmark:")
print(len(benchmark_callset.calls), "calls in benchmark callset")
print(len(benchmark_merged.starts), "calls in benchmark merged callset")
print(len(benchmark_merged_interval_set.starts),
      f"intervals after the {SIZE_FLOOR:,} bp size floor")

# 2. Query
query_callset: CallSet = collect_callsets(read_bed_calls(bed) for bed in glob.glob(str(QUERY_DIR / "*/*.bed")))
query_merged: MergedCallSet = merge_components(query_callset, min_reciprocal_overlap=0.5, min_sources=2)
query_merged_interval_set: IntervalSet = evaluation_intervals(query_merged)
print("\nQuery:")
print(len(query_callset.calls), "calls in query callset")
print(len(query_merged.starts), "calls in query merged callset")
print(len(query_merged_interval_set.starts),
      f"intervals after the {SIZE_FLOOR:,} bp size floor")
