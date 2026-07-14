"""Overlap graph over genomic interval calls, shared across pipeline stages.

A set of interval calls forms an undirected overlap graph:
    node = one call (chrom, start, end, svtype, source, sample_id)
    edge = two calls that have overlap, weighted by their reciprocal overlap fraction

Edges never cross the `(sample_id, svtype, chrom)` partition.
"""

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import networkx as nx

from consensuscnv.calls import Call


def read_bed_file(path: Path) -> list[Call]:
    """Read a per-source BED file into `Call`s.

    Expects the pipeline's 5-column layout (chrom, start, end, svtype, source).
    The sample id is the filename text before the first dot.
    """
    sample_id = path.name.split(".")[0]
    calls: list[Call] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chrom, start, end, svtype, source = line.split("\t")[:5]
            calls.append(Call(chrom, int(start), int(end), svtype, source, sample_id))
    return calls


def reciprocal_overlap(a: Call, b: Call) -> float:
    """Reciprocal overlap fraction of two calls, in [0, 1].

    Equivalent to `bedtools intersect -f <t> -r`: requiring this value >= t is
    the same as requiring the overlap to cover at least fraction t of *both*
    intervals, since dividing the shared span by the longer interval bounds both
    ratios at once.
    """
    overlap = min(a.end, b.end) - max(a.start, b.start)
    if overlap <= 0:
        return 0.0
    longest = max(a.end - a.start, b.end - b.start)
    return overlap / longest


def build_graph(
    calls: list[Call],
    *,
    link_same_source: bool = False,
    min_edge_overlap: float = 0.0,
    padding: int = 0,
) -> nx.Graph[int]:
    """Build the overlap graph from interval calls.

    Every call becomes a node (id = its index in `calls`, `call` attribute set),
    so calls with no matches still appear. Any two calls that overlap at all are
    joined by an edge whose `weight` is their `reciprocal_overlap`; no threshold
    is applied here. Choosing a threshold is left to the merge function, which
    filters edges by `weight` -- so one graph serves every threshold. Same-source
    pairs are skipped unless `link_same_source` is set.

    `padding` widens the reach for creating edges by that many bases, matching
    `bedtools merge -d <padding>` exactly: two calls are joined when the gap
    between them is `<= padding` (so `padding=0` also links book-ended calls,
    those touching end-to-start). Padded edges over a non-overlapping gap have
    `weight == 0.0`, so they only survive `merge_components(min_weight=0.0)` --
    use padding for gap-tolerant merging (e.g. benchmarks), not for reciprocal-
    overlap consensus, where any positive threshold filters them back out.

    `min_edge_overlap` is only a construction floor to keep the graph sparse on
    very large cohorts; leave it at 0.0 (keep every overlap) unless edge count
    becomes a problem, and keep it well below any threshold you sweep. It gates
    padded zero-overlap edges too, so keep it at 0.0 when using `padding`.

    Edges never cross `(sample_id, svtype, chrom)`, so calls from different
    samples stay in separate sub-graphs even when passed in together.
    """
    graph: nx.Graph[int] = nx.Graph()
    for node_id, call in enumerate(calls):
        graph.add_node(node_id, call=call)

    # Two calls can only match within the same sample, svtype, and chrom;
    # partitioning on that key enforces it and prunes the pairwise comparison.
    partitions: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for node_id, call in enumerate(calls):
        partitions[(call.sample_id, call.svtype, call.chrom)].append(node_id)

    for node_ids in partitions.values():
        node_ids.sort(key=lambda n: calls[n].start)
        for i, a_id in enumerate(node_ids):
            a: Call = calls[a_id]
            for b_id in node_ids[i + 1 :]:
                b: Call = calls[b_id]
                if b.start > a.end + padding:
                    break  # sorted by start: nothing later is within padding of `a`
                if not link_same_source and a.sources == b.sources:
                    continue
                overlap = reciprocal_overlap(a, b)
                if overlap >= min_edge_overlap:
                    graph.add_edge(a_id, b_id, weight=overlap)

    return graph


def build_sample_graphs(calls: list[Call], **kwargs) -> dict[str, nx.Graph[int]]:
    """Build one overlap graph per sample, keyed by `sample_id`."""
    by_sample: dict[str, list[Call]] = defaultdict(list)
    for call in calls:
        by_sample[call.sample_id].append(call)
    return {
        sample_id: build_graph(sample_calls, **kwargs)
        for sample_id, sample_calls in by_sample.items()
    }


def build_sample_graphs_from_beds(bed_paths: Iterable[Path], **kwargs) -> dict[str, nx.Graph[int]]:
    """Build one overlap graph per sample, keyed by `sample_id`."""
    calls = [call for path in bed_paths for call in read_bed_file(path)]
    return build_sample_graphs(calls, **kwargs)


def merge_components(
    graph: nx.Graph[int],
    *,
    min_nodes: int = 1,
    min_weight: float = 0.0,
    chrom_order: list[str] | None = None,
) -> list[Call]:
    """Merge connected components of the overlap graph into intervals.

    Edges below `min_weight` are dropped first (isolated nodes are kept), then
    every remaining connected component of at least `min_nodes` calls is merged
    into one `Call`.

    `min_nodes=1, min_weight=0.0` returns all calls merged (the union set).

    Output is sorted by genomic coordinate. `chrom_order` sets the chromosome
    ordering; contigs not in it sort after the known ones. When omitted, chroms
    sort lexicographically (`chr1, chr10, chr11, ... chr2`).
    """
    filtered: nx.Graph[int] = nx.Graph()
    filtered.add_nodes_from(graph.nodes(data=True))
    filtered.add_edges_from(
        (u, v, data) for u, v, data in graph.edges(data=True) if data["weight"] >= min_weight
    )

    merged: list[Call] = []
    for component in nx.connected_components(filtered):
        if len(component) >= min_nodes:
            merged.append(_merge_component(graph, component))

    # Sort by genomic coordinate; unknown contigs sort after the known ones.
    rank: dict[str, int] = {chrom: i for i, chrom in enumerate(chrom_order or [])}
    merged.sort(key=lambda c: (rank.get(c.chrom, len(rank)), c.chrom, c.start, c.end))
    return merged


def _merge_component(graph: nx.Graph[int], component: set[int]) -> Call:
    """Merge one component's calls into a single union-span `Call`."""
    calls: list[Call] = [
        graph.nodes[node_id]["call"]
        for node_id in component
        if type(graph.nodes[node_id]["call"]) is Call
    ]
    return Call(
        chrom=calls[0].chrom,  # uniform within a component by construction
        start=min(call.start for call in calls),
        end=max(call.end for call in calls),
        svtype=calls[0].svtype,  # uniform within a component by construction
        sources=frozenset(call.sources for call in calls),
        sample_id=calls[0].sample_id,  # uniform within a component by construction
        members=tuple(sorted(component)),
    )
