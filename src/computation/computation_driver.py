from computation.benchmark_parser import process_benchmarks_to_beds
from computation.penncnv_parser import process_penncnv_to_beds
from computation.computation_functions import run_binary_classification_script
from utils import PipelineConfig


def main(config: PipelineConfig):
    """Main computation pipeline."""
    layout = config.layout

    print("\nProcessing control datasets (SNP Array)...")
    _ = process_penncnv_to_beds(config)

    print("\nParsing all benchmarks to BED format...")
    _ = process_benchmarks_to_beds(config)

    print("\nRunning binary classification script...")
    bin_class_sets = []
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
