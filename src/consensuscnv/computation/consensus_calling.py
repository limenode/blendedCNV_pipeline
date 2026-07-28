from collections import defaultdict
import glob
from pathlib import Path

import networkx as nx

from consensuscnv.overlap_graph import (
    generate_graph_from_calls,
    merge_graph_components,
    read_bed_file,
    dump_calls_to_bed,
    merge_component,
    resolve_components
)
from consensuscnv.output_layout import (
    BenchmarkMergeParams,
    ConsensusParams,
    OutputLayout,
)
from consensuscnv.utils import PipelineConfig
from consensuscnv.calls import Call


def compute_experimental_consensus(
    config: PipelineConfig,
    weights: list[float],
) -> dict[tuple[str, float, int], list[Call]]:
    """Compute consensus calls from per-caller BED files and write them to the output folder.

    Returns ``{call_set: {source_slug: consensus_dir}}`` where ``source_slug``
    is e.g. ``'consensus_2of3_w0.5'`` — usable directly as the ``source`` leaf of
    ``layout.classification_dir``."""

    layout = config.layout
    chrom_order = config.chromosome_order
    experimental_keys = config.experimental.keys()
    _out = {}

    for experimental_key in experimental_keys:
        bed_paths_str: list[str] = glob.glob(str(layout.call_set_dir(experimental_key)) + "/*/*.bed")
        bed_paths = [Path(p) for p in bed_paths_str if Path(p).is_file()]
        
        calls = []
        for path in bed_paths:
            calls.extend(read_bed_file(path, membership=experimental_key))
        _graph = generate_graph_from_calls(calls)

        for weight in weights:
            _merged = [
                (len(call.sources), call)
                for call in (
                    merge_component(_graph, component)
                    for component in resolve_components(_graph, min_nodes=1, min_weight=weight)
                )
            ]
            _merged.sort(key=lambda pair: pair[1].sort_key(chrom_order))
            for level in [1, 2, 3]:
                _out[(experimental_key, weight, level)] = [call for n, call in _merged if n >= level]

    return _out

def load_benchmark_graph(config: PipelineConfig) -> nx.Graph:
    """Build the overlap graph over all parsed benchmark calls."""
    
    layout = config.layout
    benchmark_calls = []
    for key in config.benchmark.keys():
        for path_str in glob.glob(str(layout.benchmark_dir(key)) + "/*.bed"):
            path = Path(path_str)
            if path.is_file():
                benchmark_calls.extend(read_bed_file(path, membership=key))
    return generate_graph_from_calls(benchmark_calls)


def merge_benchmarks(
    config: PipelineConfig,
    params: BenchmarkMergeParams = BenchmarkMergeParams(),
    benchmark_graph: nx.Graph | None = None,
) -> Path:
    """Merge benchmark calls under ``params`` and write them to the output folder.

    Pass a prebuilt ``benchmark_graph`` (from :func:`load_benchmark_graph`) to reuse
    it across parameter sweeps; otherwise it is built on demand. Returns the merged
    output directory (``benchmark/merged/<bench slug>``)."""

    layout = config.layout
    graph = benchmark_graph if benchmark_graph is not None else load_benchmark_graph(config)

    merged_calls = merge_graph_components(
        graph,
        min_nodes=params.min_nodes,
        min_weight=params.min_weight,
        padding=params.padding,
        link_same_source=params.link_same_source,
    )

    output_path = layout.benchmark_merge_dir(params)

    dump_calls_to_bed(
        merged_calls,
        dir_path=output_path,
        chrom_order=config.chromosome_order,
        separate_by_sample=True,
    )

    return output_path