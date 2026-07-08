"""Consensus calling over the input-set overlap graph.

Wraps the shared overlap-graph primitives (see `overlap_graph`) with the
consensus-specific driver that writes 1/2/3-of-3 call sets to the output layout.
Each consensus level is just `merge_components` with a different `min_nodes`.
"""

import glob
from pathlib import Path

from utils import PipelineConfig
from overlap_graph import build_sample_graphs_from_beds, merge_components


def compute_consensus_from_beds(config: PipelineConfig, weight_threshold: float = 0.5):
    """Compute consensus calls from per-caller BED files and write them to the layout.

    Args:
        config (PipelineConfig): The pipeline configuration.
        weight_threshold (float, optional): The minimum weight for edges in the overlap graph. Defaults to 0.5.
    """

    layout = config.layout
    input_keys = config.experimental.keys()

    input_network_paths = {}
    for key in input_keys:
        bed_paths = glob.glob(str(layout.bed_dir(key)) + "/*/*.bed")
        input_network_paths[key] = [Path(p) for p in bed_paths if Path(p).is_file()]

    input_networks = {}

    for key in input_keys:
        input_networks[key] = build_sample_graphs_from_beds(input_network_paths[key])

    for key, networks in input_networks.items():
        for sample_id, network in networks.items():
            for level in (1, 2, 3):
                consensus_calls = merge_components(
                    network,
                    min_nodes=level,
                    min_weight=weight_threshold,
                    chrom_order=config.chromosome_order,
                )
                output_dir = layout.consensus_rep_dir(key, level, "unions")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{sample_id}.bed"
                output_file_del = output_dir / f"{sample_id}.DEL.union.bed"
                output_file_dup = output_dir / f"{sample_id}.DUP.union.bed"

                with open(output_file, "w") as f:
                    for call in consensus_calls:
                        f.write(f"{call.bed_str()}\n")

                with open(output_file_del, "w") as f:
                    for call in consensus_calls:
                        if call.svtype == "DEL":
                            f.write(f"{call.bed_str()}\n")

                with open(output_file_dup, "w") as f:
                    for call in consensus_calls:
                        if call.svtype == "DUP":
                            f.write(f"{call.bed_str()}\n")

                # Temporarily copy the union files to the "intersection" directory for downstream processing
                intersection_dir = layout.consensus_rep_dir(key, level, "intersections")
                intersection_dir.mkdir(parents=True, exist_ok=True)
                intersection_file = intersection_dir / f"{sample_id}.bed"
                intersection_file_del = intersection_dir / f"{sample_id}.DEL.intersection.bed"
                intersection_file_dup = intersection_dir / f"{sample_id}.DUP.intersection.bed"

                import shutil
                shutil.copy(output_file, intersection_file)
                shutil.copy(output_file_del, intersection_file_del)
                shutil.copy(output_file_dup, intersection_file_dup)
