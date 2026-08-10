# %% Imports and constants
"""Diagnostics and timings for consensuscnv.callsets."""

import glob
from pathlib import Path
from timeit import timeit
import numpy as np
import matplotlib.pyplot as plt

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

from consensuscnv.classification.pairs import build_candidates
from consensuscnv.classification.classify import classify, match_topology

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

# %% Create Interval Sets
from consensuscnv.classification.intervals import IntervalSet

# 1. Benchmark
benchmark_callset = collect_callsets(read_bed_calls(bed) for bed in glob.glob(str(BENCHMARK_DIR / "*/*.bed")))
benchmark_merged = merge_components(benchmark_callset, max_padding=0)
benchmark_merged_interval_set = IntervalSet.from_merged(benchmark_merged)
print("Benchmark:")
print(len(benchmark_callset.calls), "calls in benchmark callset")
print(len(benchmark_merged.starts), "calls in benchmark merged callset")
print(len(benchmark_merged_interval_set.starts), "intervals in merged interval set")

# 2. Query
query_callset: CallSet = collect_callsets(read_bed_calls(bed) for bed in glob.glob(str(QUERY_DIR / "*/*.bed")))
query_merged: MergedCallSet = merge_components(query_callset, min_reciprocal_overlap=0.5, min_sources=2)
query_merged_interval_set: IntervalSet = IntervalSet.from_merged(query_merged)
print("\nQuery:")
print(len(query_callset.calls), "calls in query callset")
print(len(query_merged.starts), "calls in query merged callset")
print(len(query_merged_interval_set.starts), "intervals in query merged interval set")

# %% Evaluate performance across classification thresholds
candidate_set = build_candidates(query_merged_interval_set, benchmark_merged_interval_set)

# (x, (precision, recall, f1)) for each threshold
classification_threshold_to_metrics: list[tuple[float, tuple[float, float, float]]] = []
# (n_pairs, TP, found, query rows with >1 partner, truth rows with >1 partner,
#  max query partners, max truth partners) for each threshold
classification_threshold_to_topology: list[tuple[int, ...]] = []

# pre-classify for validation; skip validation for subsequent thresholds to save time
_ = classify(candidate_set, min_reciprocal_overlap=0.5, validate=True)

for threshold in np.arange(0.0, 1.0, 0.001):
    classification = classify(candidate_set, min_reciprocal_overlap=threshold, validate=False)
    summary = classification.summary()
    topology = match_topology(classification)
    classification_threshold_to_metrics.append((threshold, (summary.precision, summary.recall, summary.f1)))
    classification_threshold_to_topology.append((
        topology.n_pairs,
        topology.n_matched_query,
        topology.n_matched_truth,
        topology.n_query_multi,
        topology.n_truth_multi,
        topology.max_query_partners,
        topology.max_truth_partners,
    ))

# Plot precision, recall, and F1 score as a function of threshold
thresholds, metrics = zip(*classification_threshold_to_metrics)
precisions, recalls, f1s = zip(*metrics)

# Plot each metric on their own subplot
plt.figure(figsize=(8, 10))
plt.subplot(3, 1, 1)
plt.title('Classification Performance Across Classification Reciprocal Overlap Thresholds')
plt.plot(thresholds, precisions, label='Precision', color='blue')
plt.ylabel('Precision')
plt.subplot(3, 1, 2)
plt.plot(thresholds, recalls, label='Recall', color='orange')
plt.ylabel('Recall')
plt.subplot(3, 1, 3)
plt.plot(thresholds, f1s, label='F1 Score', color='green')
plt.xlabel('Classification Reciprocal Overlap Threshold')
plt.ylabel('F1 Score')
plt.tight_layout()
plt.savefig(TEST_DIR / "classification_performance_across_classification_reciprocal_thresholds.png", dpi=300)

# %% Many-to-many structure across the same thresholds.
# Matching is many-to-many, so n_pairs, TP (counted over query rows) and found
# (counted over truth rows) are three different numbers. They converge exactly
# when the matching collapses to 1:1.
n_pairs_t, n_tp_t, n_found_t, q_multi_t, t_multi_t, max_q_t, max_t_t = zip(
    *classification_threshold_to_topology
)

plt.figure(figsize=(8, 10))
plt.subplot(3, 1, 1)
plt.title('Match Topology Across Classification Reciprocal Overlap Thresholds')
# The three coincide once the matching is 1:1, so they are drawn at decreasing
# widths to keep the overlap visible rather than hidden.
plt.plot(thresholds, n_pairs_t, label='Matched pairs', color='purple', linewidth=3)
plt.plot(thresholds, n_tp_t, label='True positives (query rows)', color='blue', linewidth=1.75)
plt.plot(thresholds, n_found_t, label='Found (truth rows)', color='orange', linewidth=1, linestyle='--')
plt.ylabel('Count')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(thresholds, q_multi_t, label='Query rows spanning >1 truth call', color='crimson')
plt.plot(thresholds, t_multi_t, label='Truth calls split across >1 query row', color='teal')
plt.ylabel('Rows with multiple partners')
plt.yscale('symlog')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(thresholds, max_q_t, label='Max truth partners of one query row', color='crimson')
plt.plot(thresholds, max_t_t, label='Max query partners of one truth call', color='teal')
plt.xlabel('Classification Reciprocal Overlap Threshold')
plt.ylabel('Max partners')
plt.yscale('symlog')
plt.legend()

plt.tight_layout()
plt.savefig(TEST_DIR / "match_topology_across_classification_reciprocal_thresholds.png", dpi=300)

# %% Evaluate classification performance across query consensus reciprocal overlap thresholds
merging_threshold_to_metrics: list[tuple[float, tuple[float, float, float]]] = []

for threshold in np.arange(0.0, 1.0, 0.01):
    query_merged_variable: MergedCallSet = merge_components(query_callset, min_reciprocal_overlap=threshold, min_sources=3)
    query_merged_interval_set_variable: IntervalSet = IntervalSet.from_merged(query_merged_variable)
    candidate_set = build_candidates(query_merged_interval_set_variable, benchmark_merged_interval_set)
    classification = classify(candidate_set, min_reciprocal_overlap=0.5, validate=False)
    summary = classification.summary()
    merging_threshold_to_metrics.append((threshold, (summary.precision, summary.recall, summary.f1)))

# Plot precision, recall, and F1 score as a function of threshold
thresholds, metrics = zip(*merging_threshold_to_metrics)
precisions, recalls, f1s = zip(*metrics)

plt.figure(figsize=(8, 10))
plt.subplot(3, 1, 1)
plt.title('Classification Performance Across Query Consensus Reciprocal Overlap Thresholds')
plt.plot(thresholds, precisions, label='Precision', color='blue')
plt.ylabel('Precision')
plt.subplot(3, 1, 2)
plt.plot(thresholds, recalls, label='Recall', color='orange')
plt.ylabel('Recall')
plt.subplot(3, 1, 3)
plt.plot(thresholds, f1s, label='F1 Score', color='green')
plt.xlabel('Query Consensus Reciprocal Overlap Threshold')
plt.ylabel('F1 Score')
plt.tight_layout()
plt.savefig(TEST_DIR / "classification_performance_across_query_consensus_thresholds.png", dpi=300)

# %% Evaluate classification performance across benchmark padding thresholds
benchmark_padding_threshold_to_metrics: list[tuple[int, tuple[float, float, float]]] = []

padding_values_log = np.concatenate(([0],np.logspace(1, 7, num=50, base=10, dtype=np.int64)))

for threhsold in padding_values_log:
    benchmark_merged_variable: MergedCallSet = merge_components(benchmark_callset, max_padding=int(threhsold))
    benchmark_merged_interval_set_variable: IntervalSet = IntervalSet.from_merged(benchmark_merged_variable)
    candidate_set = build_candidates(query_merged_interval_set, benchmark_merged_interval_set_variable)
    classification = classify(candidate_set, min_reciprocal_overlap=0.5, validate=False)
    summary = classification.summary()
    benchmark_padding_threshold_to_metrics.append((int(threhsold), (summary.precision, summary.recall, summary.f1)))

# Plot precision, recall, and F1 score as a function of threshold
thresholds_log, metrics_log = zip(*benchmark_padding_threshold_to_metrics)
precisions_log, recalls_log, f1s_log = zip(*metrics_log)



plt.figure(figsize=(8, 10))
plt.subplot(3, 1, 1)
plt.title('Classification Performance Across Benchmark Padding Thresholds')
plt.plot(thresholds_log, precisions_log, label='Precision', color='blue')
plt.xscale('log')
plt.ylabel('Precision')
plt.subplot(3, 1, 2)
plt.plot(thresholds_log, recalls_log, label='Recall', color='orange')
plt.xscale('log')
plt.ylabel('Recall')
plt.subplot(3, 1, 3)
plt.plot(thresholds_log, f1s_log, label='F1 Score', color='green')
plt.xscale('log')
plt.xlabel('Benchmark Padding Threshold')
plt.ylabel('F1 Score')
plt.tight_layout()
plt.savefig(TEST_DIR / "classification_performance_across_benchmark_padding_thresholds.png", dpi=300)
