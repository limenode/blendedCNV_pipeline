import glob
import shutil
from pathlib import Path

import networkx as nx

from consensuscnv.overlap_graph import build_sample_graphs_from_beds, merge_components
from consensuscnv.utils import PipelineConfig


def compute_consensus_from_beds(config: PipelineConfig, weight_threshold: float = 0.5):
    """Compute consensus calls from per-caller BED files and write them to the output folder."""

    layout = config.layout
    experimental_keys = config.experimental.keys()

    networks: dict[str, dict[str, nx.Graph]] = {}
    for experimental_key in experimental_keys:
        bed_paths: list[str] = glob.glob(str(layout.bed_dir(experimental_key)) + "/*/*.bed")
        networks[experimental_key] = build_sample_graphs_from_beds(
            Path(p) for p in bed_paths if Path(p).is_file()
        )

    for experimental_key, sample_graph_dict in networks.items():
        for sample_id, graph in sample_graph_dict.items():
            for level in (1, 2, 3):
                consensus_calls = merge_components(
                    graph,
                    min_nodes=level,
                    min_weight=weight_threshold,
                    chrom_order=config.chromosome_order,
                )
                output_dir = layout.consensus_rep_dir(experimental_key, level, "unions")
                output_dir.mkdir(parents=True, exist_ok=True)


                with open(output_dir / f"{sample_id}.bed", "w") as bed_file:
                    for call in consensus_calls:
                        bed_file.write(f"{call.bed_str()}\n")

                # Temporarily copy the union files to the "intersection" directory for downstream processing
                intersection_dir = layout.consensus_rep_dir(
                    experimental_key, level, "intersections"
                )
                intersection_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(output_dir / f"{sample_id}.bed", intersection_dir / f"{sample_id}.bed")
                shutil.copy(
                    output_dir / f"{sample_id}.DEL.bed", intersection_dir / f"{sample_id}.DEL.bed"
                )
                shutil.copy(
                    output_dir / f"{sample_id}.DUP.bed", intersection_dir / f"{sample_id}.DUP.bed"
                )


def merge_benchmarks(
    config: PipelineConfig,
    weight_threshold: float = 0.0,
    merge_within_set: bool = True,
    padding: int = 0,
):
    """Merge benchmark calls from per-benchmark BED files and write them to the output folder.

    `merge_within_set` controls whether overlapping calls that share a source
    (i.e. come from the same benchmark set) are merged together. When False, only
    calls from *different* benchmark sets are merged -- overlaps within a single
    set are left intact, which can leave overlapping intervals in the output.

    `padding` merges calls separated by a gap of up to `padding` bases, equivalent
    to `bedtools merge -d <padding>`. It only takes effect while `weight_threshold`
    is 0.0, since padded edges over a gap carry zero reciprocal overlap.
    """

    layout = config.layout
    benchmark_keys = config.benchmark.keys()

    network_paths = []
    for key in benchmark_keys:
        bed_paths = glob.glob(str(layout.benchmark_dir(key)) + "/*.bed")
        network_paths.extend([Path(p) for p in bed_paths if Path(p).is_file()])

    network: dict[str, nx.Graph[int]] = build_sample_graphs_from_beds(
        network_paths, link_same_source=merge_within_set, padding=padding
    )

    for sample_id, graph in network.items():
        merged_calls = merge_components(
            graph,
            min_nodes=1,
            min_weight=weight_threshold,
            chrom_order=config.chromosome_order,
        )
        output_dir = layout.benchmark_dir("merged")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{sample_id}.bed"

        # Open the output files for writing
        with open(output_file, "w") as bed_file:
            for call in merged_calls:
                bed_file.write(f"{call.bed_str()}\n")

