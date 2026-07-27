from consensuscnv.computation.consensus_calling import compute_consensus_from_beds, merge_benchmarks
from consensuscnv.output_layout import (
    BenchmarkMergeParams,
    ClassificationParams,
    ConsensusParams,
)
from consensuscnv.utils import PipelineConfig


def run_computation(config: PipelineConfig):
    """Computation pipeline."""
    layout = config.layout

    # Parameter sets that key the output tree. Widen these into loops to sweep.
    consensus_params = ConsensusParams(min_weight=0.5)
    benchmark_params = BenchmarkMergeParams(
        padding=config.benchmark_merge_padding, min_weight=0.0
    )
    classification_params = ClassificationParams(
        reciprocal_threshold=config.matching_reciprocal_threshold
    )

    print("\nRun consensus calling...")
    consensus_bed_paths = compute_consensus_from_beds(config, consensus_params)

    print("\nRun benchmark merging...")
    benchmark_bed_path = merge_benchmarks(config, benchmark_params)

    def classification_out(query: str, source: str) -> str:
        return str(
            layout.classification_dir(
                query,
                source,
                benchmark_params=benchmark_params,
                classification_params=classification_params,
            )
        )

    binary_classification_io_sets: list[tuple[str, str] | tuple[str, str, str]] = []
    for experimental_key_path_str, experimental_key_path_dict in consensus_bed_paths.items():
        for source_slug, consensus_bed_path in experimental_key_path_dict.items():
            binary_classification_io_sets.append((
                str(consensus_bed_path),
                classification_out(experimental_key_path_str, source_slug),
                str(benchmark_bed_path),
            ))

    for exp_key, tools in config.experimental.items():
        for tool in tools:
            binary_classification_io_sets.append((
                str(layout.bed_tool_dir(exp_key, tool)),
                classification_out(exp_key, tool),
                str(benchmark_bed_path),
            ))

    for ctrl_key in config.control.keys():
        binary_classification_io_sets.append((
            str(layout.control_bed_dir(ctrl_key)),
            classification_out(ctrl_key, "calls"),
            str(benchmark_bed_path),
        ))

    print("Binary classification I/O sets:")
    for query_path, output_path, *truth_dir in binary_classification_io_sets:
        if truth_dir:
            print(f"Query: {query_path}, Output: {output_path}, Truth: {truth_dir[0]}")
        else:
            print(f"Query: {query_path}, Output: {output_path}, Truth: None")

    # run_binary_classification_script(
    #     config,
    #     binary_classification_io_sets,
    #     benchmark_bed_path,
    #     reciprocal_threshold=classification_params.reciprocal_threshold,
    # )
