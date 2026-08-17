# %% Imports and constants

import glob
from pathlib import Path
import itertools
import numpy as np
import matplotlib.pyplot as plt

from consensuscnv.callsets import (
    MergedCallSet,
    collect_callsets,
    merge_components,
    read_bed_calls,
)

from consensuscnv.classification.pairs import build_candidates
from consensuscnv.classification.classify import classify, match_topology
from consensuscnv.classification.intervals import IntervalSet

BENCHMARK_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark")
CONTROL_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/SNP_Array")
QUERY_DIRS = {
    "30x": Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/30x_Coverage"),
    "6x": Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/6x_Coverage"),
    "4x": Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/4x_Coverage"),
    "2x": Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/2x_Coverage"),
}
QUERY_DIR = QUERY_DIRS["30x"]
RESULTS_DIR = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/results")

# The detectable size domain.
SIZE_FLOOR = 1_000


def evaluation_intervals(merged: MergedCallSet) -> IntervalSet:
    """Merged calls as an IntervalSet, restricted to the detectable size domain."""
    return IntervalSet.from_merged(merged).filter_by_size(min_size=SIZE_FLOOR)


# %% Collect call sets
benchmark_callset = collect_callsets(read_bed_calls(bed) for bed in glob.glob(str(BENCHMARK_DIR / "*/*.bed")))
query_callset = collect_callsets(read_bed_calls(bed) for bed in glob.glob(str(QUERY_DIR / "*/*.bed")))

# %% Merge call sets and create interval sets
benchmark_merged_interval_set = evaluation_intervals(merge_components(benchmark_callset, max_padding=0))
query_merged_interval_set = evaluation_intervals(merge_components(query_callset, min_reciprocal_overlap=0.5, min_sources=2))

# %% Evaluate performance across classification thresholds
candidate_set = build_candidates(query_merged_interval_set, benchmark_merged_interval_set)

# (x, (precision, recall, f1)) for each threshold
classification_threshold_to_metrics: list[tuple[float, tuple[float, float, float]]] = []
# (n_pairs, TP, found, query rows with >1 partner, truth rows with >1 partner, max query partners, max truth partners) for each threshold
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
plt.savefig(RESULTS_DIR / "classification_performance_across_classification_reciprocal_thresholds.png", dpi=300)

# %% Many-to-many structure across the same thresholds.
# Matching is many-to-many, so n_pairs, TP (counted over query rows) and found
# (counted over truth rows) are three different numbers. They converge exactly
# when the matching collapses to 1:1.
n_pairs_t, n_tp_t, n_found_t, q_multi_t, t_multi_t, max_q_t, max_t_t = zip(
    *classification_threshold_to_topology
)

print("Max query rows with multiple partners across thresholds:", max(q_multi_t))
print("Max truth calls with multiple partners across thresholds:", max(t_multi_t))

print("Max query partners across thresholds:", max(max_q_t))
print("Max truth partners across thresholds:", max(max_t_t))


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
plt.savefig(RESULTS_DIR / "match_topology_across_classification_reciprocal_thresholds.png", dpi=300)

# %% Evaluate classification performance across query consensus reciprocal overlap thresholds
merging_threshold_to_metrics: list[tuple[float, tuple[float, float, float]]] = []

for threshold in np.arange(0.0, 1.0, 0.01):
    query_merged_variable: MergedCallSet = merge_components(query_callset, min_reciprocal_overlap=threshold, min_sources=3)
    query_merged_interval_set_variable: IntervalSet = evaluation_intervals(query_merged_variable)
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
plt.savefig(RESULTS_DIR / "classification_performance_across_query_consensus_thresholds.png", dpi=300)

# %% Evaluate classification performance across benchmark padding thresholds
benchmark_padding_threshold_to_metrics: list[tuple[int, tuple[float, float, float]]] = []

padding_values_log = np.concatenate(([0],np.logspace(1, 7, num=50, base=10, dtype=np.int64)))

_query_merged_variable: MergedCallSet = merge_components(query_callset, min_reciprocal_overlap=0.5, min_sources=2)
_query_merged_interval_set_variable: IntervalSet = evaluation_intervals(_query_merged_variable)

for threhsold in padding_values_log:
    benchmark_merged_variable: MergedCallSet = merge_components(benchmark_callset, max_padding=int(threhsold))
    benchmark_merged_interval_set_variable: IntervalSet = evaluation_intervals(benchmark_merged_variable)
    candidate_set = build_candidates(_query_merged_interval_set_variable, benchmark_merged_interval_set_variable)
    classification = classify(candidate_set, min_reciprocal_overlap=0.3, validate=False)
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
plt.savefig(RESULTS_DIR / "classification_performance_across_benchmark_padding_thresholds.png", dpi=300)

# %% Classification performance as a function of CNV size
from consensuscnv.classification.classify import (
    SizeBinning,
    size_density_curve,
    size_metrics,
)
from consensuscnv.utils import DistributionType

# Rebuilt set
size_candidate_set = build_candidates(query_merged_interval_set, benchmark_merged_interval_set)
size_classification = classify(size_candidate_set, min_reciprocal_overlap=0.5, validate=False)

size_binning = SizeBinning.at_every_size(size_candidate_set)
size_ccdf = size_metrics(
    size_classification, size_binning, distribution=DistributionType.COMPLEMENTARY_CUMULATIVE
)
size_cdf = size_metrics(
    size_classification, size_binning, distribution=DistributionType.CUMULATIVE
)
print(f"{len(size_binning)} distinct sizes across "
      f"{len(query_merged_interval_set) + len(benchmark_merged_interval_set)} rows")

size_x = size_binning.edges

plt.figure(figsize=(8, 12))
plt.subplot(4, 1, 1)
plt.title('Classification Performance vs CNV Size (cumulative)')
plt.plot(size_x, size_ccdf.precision[1:], label='Calls >= size', color='blue')
plt.plot(size_x, size_cdf.precision[:-1], label='Calls < size', color='blue', linestyle='--')
plt.xscale('log')
plt.ylabel('Precision')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(size_x, size_ccdf.recall[1:], label='Calls >= size', color='orange')
plt.plot(size_x, size_cdf.recall[:-1], label='Calls < size', color='orange', linestyle='--')
plt.xscale('log')
plt.ylabel('Recall')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(size_x, size_ccdf.f1[1:], label='Calls >= size', color='green')
plt.plot(size_x, size_cdf.f1[:-1], label='Calls < size', color='green', linestyle='--')
plt.xscale('log')
plt.ylabel('F1 Score')
plt.legend()


plt.subplot(4, 1, 4)
plt.plot(size_x, size_ccdf.n_query[1:], label='Query calls >= size', color='crimson')
plt.plot(size_x, size_ccdf.n_truth[1:], label='Truth calls >= size', color='teal')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('CNV Size (bp)')
plt.ylabel('Calls in denominator')
plt.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "classification_performance_across_cnv_size_cumulative.png", dpi=300)

# %% Kernel-smoothed performance vs CNV size
size_kde = size_density_curve(
    size_classification, bandwidth=0.12, n_points=512, min_effective_count=0.0
)

# Hard-binned estimate of the same quantity, for comparison.
size_density_bins = SizeBinning.from_candidates(
    size_candidate_set, np.logspace(3, 7, 30).astype(np.int64)
)
size_density = size_metrics(size_classification, size_density_bins)
size_bin_centres = np.sqrt(
    size_density_bins.lower_edges[1:-1] * size_density_bins.lower_edges[2:]
)

plt.figure(figsize=(8, 12))
plt.subplot(3, 1, 1)
plt.title(f'Classification Performance vs CNV Size '
          f'(kernel-smoothed, bandwidth {size_kde.bandwidth} dex)')
plt.plot(size_kde.sizes, size_kde.precision, label='Kernel-smoothed', color='blue')
plt.plot(size_bin_centres, size_density.precision[1:-1], label='Binned', color='blue',
         linestyle='none', marker='o', markersize=3, alpha=0.5)
plt.xscale('log')
plt.ylabel('Precision')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(size_kde.sizes, size_kde.recall, label='Kernel-smoothed', color='orange')
plt.plot(size_bin_centres, size_density.recall[1:-1], label='Binned', color='orange',
         linestyle='none', marker='o', markersize=3, alpha=0.5)
plt.xscale('log')
plt.ylabel('Recall')
plt.legend()

# The size distributions themselves -- these are the only curves here that
# integrate to one. Where they are thin, the metrics above are masked out.
plt.subplot(3, 1, 3)
plt.plot(size_kde.sizes, size_kde.query_density, label='Query calls', color='crimson')
plt.plot(size_kde.sizes, size_kde.truth_density, label='Truth calls', color='teal')
plt.xscale('log')
plt.xlabel('CNV Size (bp)')
plt.ylabel('Density (per log10 bp)')
plt.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "classification_performance_across_cnv_size_density.png", dpi=300)

# %%
range_of_benchmark_padding = np.array([0, 10, 25, 50, 100, 200, 400, 700, 1000, 5000, 10000])
range_of_consensus_threhsolds = np.round(np.arange(0.05, 1.0, 0.05), 2)
range_of_classify_threshold = np.round(np.arange(0.05, 1.0, 0.05), 2)

all_combinations = list(itertools.product(range_of_consensus_threhsolds, range_of_benchmark_padding, range_of_classify_threshold))

query_callsets_dict = {
    threshold: evaluation_intervals(merge_components(query_callset, min_reciprocal_overlap=threshold, min_sources=2))
    for threshold in range_of_consensus_threhsolds
}

benchmark_callsets_dict = {
    padding: evaluation_intervals(merge_components(benchmark_callset, max_padding=int(padding)))
    for padding in range_of_benchmark_padding
}

candidate_sets = {
    (consensus_threshold, benchmark_padding): build_candidates(
        query_callsets_dict[consensus_threshold],
        benchmark_callsets_dict[int(benchmark_padding)]
    )
    for consensus_threshold, benchmark_padding in itertools.product(range_of_consensus_threhsolds, range_of_benchmark_padding)
}

# %%
shape = (len(range_of_consensus_threhsolds), len(range_of_benchmark_padding), len(range_of_classify_threshold))
precision = np.empty(shape)
recall = np.empty(shape)
f1 = np.empty(shape)
n_truth = np.empty(shape, dtype=np.int64)

for (i, c), (j, p) in itertools.product(enumerate(range_of_consensus_threhsolds), enumerate(range_of_benchmark_padding)):
    cand = candidate_sets[(c, int(p))]
    for k, m in enumerate(range_of_classify_threshold):
        classification = classify(cand, min_reciprocal_overlap=m, validate=False)
        summary = classification.summary()
        precision[i, j, k] = summary.precision
        recall[i, j, k] = summary.recall
        f1[i, j, k] = summary.f1
        n_truth[i, j, k] = len(classification.truth)


# %%

# Get parameters that correspond to the best metric scores
precision_index = np.unravel_index(np.nanargmax(precision), precision.shape)
recall_index = np.unravel_index(np.nanargmax(recall), recall.shape)
f1_index = np.unravel_index(np.nanargmax(f1), f1.shape)

print(f"Best precision: {precision[precision_index]:.4f} at \
    consensus threshold {range_of_consensus_threhsolds[precision_index[0]]:.2f}, \
    benchmark padding {range_of_benchmark_padding[precision_index[1]]}, \
    classification threshold {range_of_classify_threshold[precision_index[2]]:.2f}")
print(f"Best recall: {recall[recall_index]:.4f} at \
    consensus threshold {range_of_consensus_threhsolds[recall_index[0]]:.2f}, \
    benchmark padding {range_of_benchmark_padding[recall_index[1]]}, \
    classification threshold {range_of_classify_threshold[recall_index[2]]:.2f}")
print(f"Best F1: {f1[f1_index]:.4f} at \
    consensus threshold {range_of_consensus_threhsolds[f1_index[0]]:.2f}, \
    benchmark padding {range_of_benchmark_padding[f1_index[1]]}, \
    classification threshold {range_of_classify_threshold[f1_index[2]]:.2f}")

# %% Sensitivity analysis: which parameter drives the variation?
from consensuscnv.analysis.sobol import decompose, sobol_indices, write_sensitivity_tsv

# Axis order must match `shape` above.
FACTOR_NAMES = ("consensus threshold", "benchmark padding", "classify threshold")

# The additive model: grand mean plus one curve per parameter. Its R^2 is what
# licenses (or refuses) the one-parameter marginal plots below.
f1_model = decompose(f1)
print(f"Grand mean: {f1_model.grand:.4f}")
print(f"Fraction of variance due to interaction: {f1_model.interaction_fraction:.4f}")
print(f"Additive model R^2: {f1_model.r_squared:.4f}, from "
      f"{f1_model.n_free_parameters} free parameters over {f1.size} cells")

# Report precision and recall separately, not just F1: with precision >> recall
# throughout this grid, F1 = 2PR/(P+R) collapses towards 2R and its indices
# mostly restate recall's.
sensitivity = {
    metric_name: sobol_indices(field, FACTOR_NAMES)
    for metric_name, field in (("f1", f1), ("precision", precision), ("recall", recall))
}
for metric_name, indices in sensitivity.items():
    print(f"{metric_name}: {indices}")

n_rows = write_sensitivity_tsv(RESULTS_DIR / "sobol_indices.tsv", sensitivity)
print(f"wrote {n_rows} rows to {RESULTS_DIR / 'sobol_indices.tsv'}")

# %% Precision/recall Pareto front
pf, rf = precision.ravel(), recall.ravel()
order = np.argsort(-rf)
front, best = [], -np.inf
for idx in order:
    if pf[idx] > best:
        front.append(idx)
        best = pf[idx]
