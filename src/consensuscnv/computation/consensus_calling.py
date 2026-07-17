import glob
from pathlib import Path

import networkx as nx

from consensuscnv.calls import Call
from consensuscnv.overlap_graph import (
    generate_graph_from_calls,
    merge_component,
    merge_graph_components,
    read_bed_file,
    split_calls_by_svtype,
    sort_calls,
    dump_calls_to_bed,
)
from consensuscnv.utils import PipelineConfig


def compute_consensus_from_beds(
    config: PipelineConfig, 
    weight_threshold: float = 0.5
) -> None:
    """Compute consensus calls from per-caller BED files and write them to the output folder."""

    layout = config.layout
    experimental_keys = config.experimental.keys()
    
    for experimental_key in experimental_keys:
        bed_paths_str: list[str] = glob.glob(str(layout.bed_dir(experimental_key)) + "/*/*.bed")
        bed_paths = [Path(p) for p in bed_paths_str if Path(p).is_file()]
        calls = []
        for path in bed_paths:
            calls.extend(read_bed_file(path, membership=experimental_key))
        graph = generate_graph_from_calls(calls)
        for level in [1, 2, 3]:
            merged_calls = merge_graph_components(
                graph,
                min_nodes=level,
                min_weight=weight_threshold,
            )
            dump_calls_to_bed(
                merged_calls,
                dir_path=Path(layout.consensus_rep_dir(experimental_key, level, "intersections")),
                chrom_order=config.chromosome_order
            )


def merge_benchmarks(
    config: PipelineConfig,
    min_nodes: int = 1,
    weight_threshold: float = 0.0,
    merge_within_set: bool = True,
    padding: int = 0,
) -> None:
    """Merge benchmark calls from per-benchmark BED files and write them to the output folder."""

    layout = config.layout
    benchmark_keys = config.benchmark.keys()

    benchmark_calls = []
    for _key in benchmark_keys:
        bed_paths_test = glob.glob(str(layout.benchmark_dir(_key)) + "/*.bed")
        for _path in bed_paths_test:
            if Path(_path).is_file():
                benchmark_calls.extend(read_bed_file(Path(_path), membership=_key))

    benchmark_graph = generate_graph_from_calls(benchmark_calls)
    
    merged_calls = merge_graph_components(
        benchmark_graph,
        min_nodes=min_nodes,
        min_weight=weight_threshold,
        padding=padding,
        link_same_source=merge_within_set
    )
    
    dump_calls_to_bed(
        merged_calls,
        dir_path=layout.benchmark_dir("merged"),
        chrom_order=config.chromosome_order,
        separate_by_sample=True,
    )
    