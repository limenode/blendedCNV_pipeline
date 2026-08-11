# %% Imports and constants

from pathlib import Path
from timeit import timeit

from consensuscnv.callsets import (
    build_callset,
    collect_callsets,
    filter_edges,
    merge_components,
    read_bed_calls,
    write_merged_bed,
)

BENCHMARK_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark")
QUERY_DIR: Path = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/30x_Coverage")
TEST_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/src/test")

# %% Test collecting benchmark call sets for HG00096
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

aggregate_benchmark_hg00096 = collect_callsets([cs1, cs2, cs3])
print(len(aggregate_benchmark_hg00096.calls), "calls in merged callset.")
print(len(aggregate_benchmark_hg00096.ov_key), "edges in merged callset.")

# %% Merge components into merged calls
print("merge_components (HG00096) timeit:", timeit(
    lambda: merge_components(aggregate_benchmark_hg00096, max_padding=0, min_calls=1, min_sources=1),
    number=10,
))

merged_set = merge_components(aggregate_benchmark_hg00096, max_padding=0, min_calls=1, min_sources=1)
print("HG00096 Benchmark Merged Calls Total:", len(merged_set.starts))

print("write_merged_bed (HG00096) timeit:", timeit(
    lambda: write_merged_bed(merged_set, TEST_DIR / "output.bed"),
    number=1,
))

# %% Consensus callset based on source counts
selection = filter_edges(aggregate_benchmark_hg00096, max_padding=0)
merged = merge_components(aggregate_benchmark_hg00096, selection)
for level in (1, 2, 3):
    print(f">= {level} source(s):", int((merged.n_sources >= level).sum()))
