from consensuscnv.computation.computation_functions import run_binary_classification_script
from consensuscnv.computation.consensus_calling import compute_consensus_from_beds, merge_benchmarks
from consensuscnv.utils import PipelineConfig


def run_computation(config: PipelineConfig):
    """Computation pipeline."""
    layout = config.layout

    print("\nRun consensus calling...")
    consensus_bed_paths = compute_consensus_from_beds(config, weight_threshold=0.5)

    print("\nRun benchmark merging...")
    benchmark_bed_path_p0 = merge_benchmarks(config, weight_threshold=0.0, padding=config.benchmark_merge_padding)

    binary_classification_io_sets: list[tuple[str, str] | tuple[str, str, str]] = []
    for experimental_key_path_str, experimental_key_path_dict in consensus_bed_paths.items():
        for consensus_level, consensus_bed_path in experimental_key_path_dict.items():
            
            binary_classification_io_sets.append((
                str(consensus_bed_path),
                str(layout.classification_dir("padding0",experimental_key_path_str, consensus_level)),
                str(benchmark_bed_path_p0)
            ))

    for exp_key, tools in config.experimental.items():
        for tool in tools:
            binary_classification_io_sets.append((
                str(layout.bed_tool_dir(exp_key, tool)),
                str(layout.classification_dir("padding0", exp_key, tool))
            ))

    for ctrl_key in config.control.keys():
        binary_classification_io_sets.append(
            (
                str(layout.control_bed_dir(ctrl_key)),
                str(layout.classification_dir("padding0", ctrl_key, "calls")),
            )
        )

    print("Binary classification I/O sets:")
    for input_path, output_path, *truth_dir in binary_classification_io_sets:
        if truth_dir:
            print(f"Input: {input_path}, Output: {output_path}, Truth: {truth_dir[0]}")
        else:
            print(f"Input: {input_path}, Output: {output_path}, Truth: None")

    run_binary_classification_script(config, binary_classification_io_sets, benchmark_bed_path_p0)
