from consensuscnv.computation.computation_functions import run_binary_classification_script
from consensuscnv.computation.consensus_calling import compute_consensus_from_beds, merge_benchmarks
from consensuscnv.utils import PipelineConfig


def run_computation(config: PipelineConfig):
    """Computation pipeline."""
    layout = config.layout

    print("\nRun consensus calling...")
    compute_consensus_from_beds(config, weight_threshold=0.5)

    print("\nRun benchmark merging...")
    merge_benchmarks(config, weight_threshold=0.0, padding=config.benchmark_merge_padding)

    print("\nRunning binary classification script...")
    bin_class_sets: list[tuple[str, str]] = []
    for key, tools in config.experimental.items():
        # Add the consensus call sets (intersections, then unions) for classification
        for representation in ("intersections", "unions"):
            for level in (1, 2, 3):
                call_set = layout.consensus_call_set(level, representation)
                bin_class_sets.append(
                    (
                        str(layout.consensus_rep_dir(key, level, representation)),
                        str(layout.classification_dir(key, call_set)),
                    )
                )

        # Add individual tool results to the sets for classification
        for tool in tools.keys():
            bin_class_sets.append(
                (
                    str(layout.bed_tool_dir(key, tool)),
                    str(layout.classification_dir(key, tool)),
                )
            )

    # Add control datasets to the sets for classification
    for key in config.control.keys():
        bin_class_sets.append(
            (
                str(layout.control_bed_dir(key)),
                str(layout.control_classification_dir(key)),
            )
        )

    run_binary_classification_script(config, bin_class_sets)
