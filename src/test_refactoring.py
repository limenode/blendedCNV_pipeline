# %%
"""Diagnostics and timings for consensuscnv.callsets.

The functions and types this exercises now live in the package -- this file is
only the scratch surface for timing them and eyeballing results.
"""

import glob
from pathlib import Path
from timeit import timeit

from consensuscnv.callsets import (
    Call,
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
def retrieve_overlap(call_a: Call, call_b: Call) -> tuple[bool, float, int]:
    """Pairwise reference implementation, kept as an oracle for the sweep in
    `build_callset`. Not used by the package."""
    reciprocal_overlap = 0.0
    distance = 0

    if call_a.chrom != call_b.chrom:
        return False, 0, 0

    overlap_start = max(call_a.start, call_b.start)
    overlap_end = min(call_a.end, call_b.end)

    if overlap_start < overlap_end:
        overlap_length = overlap_end - overlap_start
        reciprocal_overlap = overlap_length / max(
            call_a.end - call_a.start, call_b.end - call_b.start
        )
        distance = 0
    else:
        distance = min(abs(call_a.start - call_b.end), abs(call_b.start - call_a.end))

    return True, reciprocal_overlap, distance


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
print(timeit(lambda: collect_callsets([cs1, cs2, cs3]), number=1))

merged_callset = collect_callsets([cs1, cs2, cs3])
print(len(merged_callset.calls), "calls in merged callset.")
print(len(merged_callset.ov_key), "edges in merged callset.")

# %%
print(timeit(
    lambda: merge_components(merged_callset, max_padding=0, min_calls=1, min_sources=1),
    number=10,
))
merged_set = merge_components(merged_callset, max_padding=0, min_calls=1, min_sources=1)
print(len(merged_set.starts))

# %%
print(timeit(
    lambda: write_merged_bed(merged_callset, merged_set, TEST_DIR / "output.bed"),
    number=1,
))

# %%
list_of_beds = glob.glob(str(BENCHMARK_DIR / "*/*.bed"))
all_bm_callset = collect_callsets(read_bed_calls(bed) for bed in list_of_beds)
print(len(all_bm_callset.calls), "calls across", len(list_of_beds), "files")

# %%
print(timeit(
    lambda: write_merged_bed(
        all_bm_callset,
        merge_components(all_bm_callset, max_padding=0),
        TEST_DIR / "merged_all_benchmarks.bed",
        include_sample=True,
    ),
    number=1,
))

# %%
# consensus levels at a single parameter point
selection = filter_edges(merged_callset, min_reciprocal_overlap=0.5)
merged = merge_components(merged_callset, selection)
for level in (1, 2, 3):
    print(f">= {level} source(s):", int((merged.n_sources >= level).sum()))
