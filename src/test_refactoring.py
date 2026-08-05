# %%
"""Diagnostics and timings for consensuscnv.callsets."""

import glob
from pathlib import Path
from timeit import timeit
import time
from dataclasses import dataclass
import numpy as np

from consensuscnv.callsets import (
    CallSet,
    MergedCallSet,
    build_callset,
    collect_callsets,
    filter_edges,
    merge_components,
    read_bed_calls,
    write_merged_bed,
)

BENCHMARK_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark")
TEST_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/src/test")

# %%
print(timeit(lambda: build_callset(read_bed_calls(BENCHMARK_DIR / "1000G/HG00096.bed")), number=10))

# %%
cs1 = build_callset(read_bed_calls(BENCHMARK_DIR / "1000G/HG00096.bed"))
cs2 = build_callset(read_bed_calls(BENCHMARK_DIR / "HGSVC3/HG00096.bed"))
cs3 = build_callset(read_bed_calls(BENCHMARK_DIR / "ont_vienna/HG00096.bed"))
print(len(cs1.calls), "calls in callset 1; ", len(cs1.ov_key), "overlap edges.")
print(len(cs2.calls), "calls in callset 2; ", len(cs2.ov_key), "overlap edges.")
print(len(cs3.calls), "calls in callset 3; ", len(cs3.ov_key), "overlap edges.")
print(
    len(cs1.calls) + len(cs2.calls) + len(cs3.calls),
    "calls in total; ",
    len(cs1.ov_key) + len(cs2.ov_key) + len(cs3.ov_key),
    "overlap edges.",
)
print("collect_callsets (HG00096)", timeit(lambda: collect_callsets([cs1, cs2, cs3]), number=1))

merged_callset = collect_callsets([cs1, cs2, cs3])
print(len(merged_callset.calls), "calls in merged callset.")
print(len(merged_callset.ov_key), "edges in merged callset.")

# %%
print("merge_components (HG00096):", timeit(
    lambda: merge_components(merged_callset, max_padding=0, min_calls=1, min_sources=1),
    number=10,
))

merged_set = merge_components(merged_callset, max_padding=0, min_calls=1, min_sources=1)
print("HG00096 Benchmark Merged Calls Total:", len(merged_set.starts))

print("write_merged_bed (HG00096):", timeit(
    lambda: write_merged_bed(merged_set, TEST_DIR / "output.bed"),
    number=1,
))


# %%
list_of_beds = glob.glob(str(BENCHMARK_DIR / "*/*.bed"))
t1 = time.time()
all_bm_callset = collect_callsets(read_bed_calls(bed) for bed in list_of_beds)
t2 = time.time()
write_merged_bed(
    merge_components(all_bm_callset, max_padding=0),
    TEST_DIR / "merged_all_benchmarks.bed",
    include_sample=True,
)
t3 = time.time()

print(len(all_bm_callset.calls), "calls across", len(list_of_beds), "files")
print("collect_callsets (all):", t2 - t1, "seconds")
print("merge_components (all):", t3 - t2, "seconds")

# %%
# consensus levels at a single parameter point
selection = filter_edges(merged_callset, min_reciprocal_overlap=0.5)
merged = merge_components(merged_callset, selection)
for level in (1, 2, 3):
    print(f">= {level} source(s):", int((merged.n_sources >= level).sum()))
# %%
@dataclass(frozen=True, slots=True)
class IntervalSet:
    """One side of a classification: flat intervals plus their partition columns."""

    starts: np.ndarray
    ends: np.ndarray
    chrom_idx: np.ndarray
    svtype_idx: np.ndarray
    sample_idx: np.ndarray

    origin: CallSet
    row_index: np.ndarray

    def __len__(self) -> int:
        return len(self.starts)

    @classmethod
    def from_callset(cls, callset: CallSet) -> "IntervalSet":
        return cls(
            callset.starts,
            callset.ends,
            callset.chrom_idx,
            callset.svtype_idx,
            callset.sample_idx,
            callset,
            np.arange(len(callset.calls))
        )

    @classmethod
    def from_merged(cls, merged: MergedCallSet) -> "IntervalSet":
        parent = merged.parent
        rep = merged.representative
        return cls(
            merged.starts,
            merged.ends,
            parent.chrom_idx[rep],
            parent.svtype_idx[rep],
            parent.sample_idx[rep],
            parent,
            rep
        )

# %%
@dataclass(frozen=True, slots=True)
class PairSet:
    query_row: np.ndarray
    truth_row: np.ndarray
    reciprocal_overlap: np.ndarray
    distance: np.ndarray

@dataclass(frozen=True, slots=True)
class PreparedTruth:
    truth: IntervalSet
    order: np.ndarray
    start_c: np.ndarray
    reach: np.ndarray
    query_key: np.ndarray
    scale: int

@dataclass(frozen=True, slots=True)
class CandidateSet:
    # overlapping pairs, sorted by reciprocal overlap
    ov_q: np.ndarray
    ov_t: np.ndarray
    ov_key: np.ndarray

    # non-overlapping pairs, sorted by distance
    gap_q: np.ndarray
    gap_t: np.ndarray
    gap_key: np.ndarray

    built_with_padding: int


def filter_candidates(
    c,
    *,
    min_reciprocal_overlap: float = 0.0,
    max_padding: int | None = None
):
    i = np.searchsorted(c.ov_key, min_reciprocal_overlap, side='left')
    if max_padding is None:
        return c.ov_q[i:], c.ov_t[i:]

    j = np.searchsorted(c.gap_key, max_padding, side='right')
    return (
        np.concatenate((c.ov_q[i:], c.gap_q[:j])),
        np.concatenate((c.ov_t[i:], c.gap_t[:j]))
    )
