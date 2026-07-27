"""Overlap graph over genomic interval calls, shared across pipeline stages.
Edges never cross the `(sample_id, svtype, chrom)` partition.
"""

from collections import defaultdict
import os
from pathlib import Path
from typing import Iterable

import networkx as nx

from consensuscnv.calls import Call


def read_bed_file(path: Path, membership: str = "") -> list[Call]:
    """Read a per-source BED file into `Call`s.

    Expects the pipeline's 5-column layout (chrom, start, end, svtype, source).
    The sample id is the filename text before the first dot. `membership` records
    the originating set (e.g. the experimental or benchmark key).
    """
    sample_id = path.name.split(".")[0]
    calls: list[Call] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chrom, start, end, svtype, sources = line.split("\t")[:5]
            sources = frozenset(sources.split("|"))
            calls.append(Call(chrom, int(start), int(end), svtype, sources, sample_id, membership))
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


def generate_graph_from_calls(
    calls: list[Call],
) -> nx.Graph:
    """Generate a graph of calls, with edges weighted by reciprocal overlap and distance.
    Partitions are created by (sample_id, svtype, chrom) so that edges never cross these boundaries."""

    graph: nx.Graph = nx.Graph()

    # Two calls can only match within the same (sample, svtype, chrom);
    partitions: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    for node_id, call in enumerate(calls):
        graph.add_node(node_id, call=call)
        partitions[(call.sample_id, call.svtype, call.chrom)].append(node_id)

    for node_ids in partitions.values():        
        node_ids.sort(key=lambda n: calls[n].start)

        for index, a_id in enumerate(node_ids):
            a = calls[a_id]
            for j in range(index + 1, len(node_ids)):
                b_id = node_ids[j]
                b = calls[b_id]

                if b.start > a.end:
                    graph.add_edge(a_id, b_id, weight=0.0, distance=b.start - a.end)
                    break
                graph.add_edge(a_id, b_id, weight=reciprocal_overlap(a, b), distance=0)

    return graph


def resolve_graph(
    graph: nx.Graph,
    min_nodes: int = 1,
    min_weight: float = 0.0,
    padding: int = 0,
    link_same_source: bool = False,
) -> nx.Graph:
    """Resolve the graph into a subgraph of connected components that criteria."""

    if min_weight > 0 and padding > 0:
        raise ValueError("Cannot use both min_weight and padding at the same time.")

    def keep_edge(u: int, v: int, data: dict) -> bool:
        if data.get("weight", 0) < min_weight or data.get("distance", 0) > padding:
            return False
        if not link_same_source:
            return graph.nodes[u]["call"].sources != graph.nodes[v]["call"].sources
        return True

    filtered = nx.Graph()
    filtered.add_nodes_from(graph.nodes(data=True))
    filtered.add_edges_from(
        (u, v, data) for u, v, data in graph.edges(data=True) if keep_edge(u, v, data)
    )

    if min_nodes <= 1:
        return filtered

    keep_nodes = {
        node
        for component in nx.connected_components(filtered)
        if len(component) >= min_nodes
        for node in component
    }

    return filtered.subgraph(keep_nodes)


def resolve_components(
    graph: nx.Graph,
    min_nodes: int = 1,
    min_weight: float = 0.0,
    padding: int = 0,
    link_same_source: bool = False,
) -> list[set[int]]:
    """Group nodes into connected components under the given edge criteria."""
    if min_weight > 0 and padding > 0:
        raise ValueError("Cannot use both min_weight and padding at the same time.")

    union_find = nx.utils.UnionFind(graph.nodes)
    for u, v, data in graph.edges(data=True):
        if (
            data.get("weight", 0) > 0.0 and data.get("weight", 0) < min_weight 
            or data.get("distance", 0) > padding
        ):
            continue
        if (
            not link_same_source
            and graph.nodes[u]["call"].sources == graph.nodes[v]["call"].sources
        ):
            continue
        union_find.union(u, v)
    
    if min_nodes <= 1:
        return list(union_find.to_sets())

    return [component for component in union_find.to_sets() if len(component) >= min_nodes]


def merge_component(graph: nx.Graph, component: set[int]) -> Call:
    """Merge one component's calls into a single union-span `Call`."""
    if len(component) == 0:
        raise ValueError("Cannot merge an empty component.")

    if len(component) == 1:
        node_id = next(iter(component))
        return graph.nodes[node_id]["call"]
    
    # If there are multiple calls in the component, we need to merge them.
    
    # Get first call to check uniformity of chrom, svtype, and sample_id
    first_call = graph.nodes[next(iter(component))]["call"]
    start = first_call.start
    end = first_call.end
    sources = set(first_call.sources)
    memberships = set([first_call.membership]) if first_call.membership else set()
    
    for node_id in component:
        current_call = graph.nodes[node_id]["call"]
        
        if type(current_call) is not Call:
            raise ValueError(f"Node {node_id} does not contain a Call object.")
        
        if current_call.chrom != first_call.chrom:
            raise ValueError(f"Component contains calls with different chromosomes: {first_call.chrom} and {current_call.chrom}.")
        
        if current_call.svtype != first_call.svtype:
            raise ValueError(f"Component contains calls with different svtypes: {first_call.svtype} and {current_call.svtype}.")
    
        if current_call.sample_id != first_call.sample_id:
            raise ValueError(f"Component contains calls with different sample_ids: {first_call.sample_id} and {current_call.sample_id}.")
        
        start = min(start, current_call.start)
        end = max(end, current_call.end)
        sources.update(current_call.sources)
        if current_call.membership:
            memberships.add(current_call.membership)
        
    return Call(
        chrom=first_call.chrom,
        start=start,
        end=end,
        svtype=first_call.svtype,
        sources=frozenset(sources),
        sample_id=first_call.sample_id,
        membership="|".join(sorted(memberships)) if memberships else "",
    )


def merge_graph_components(
    graph: nx.Graph,
    min_nodes: int = 1,
    min_weight: float = 0.0,
    padding: int = 0,
    link_same_source: bool = False,
) -> list[Call]:
    """Merge all components of a graph into union-span `Call`s."""
    components = resolve_components(graph, min_nodes=min_nodes, min_weight=min_weight, padding=padding, link_same_source=link_same_source)
    return [merge_component(graph, component) for component in components]


def sort_calls(calls: Iterable[Call], chrom_order: list[str] | None) -> list[Call]:
    """Sort calls by (chrom, start, end, svtype, sample_id)."""

    # Sort by genomic coordinate; unknown contigs sort after the known ones.
    rank: dict[str, int] = {chrom: i for i, chrom in enumerate(chrom_order or [])}
    ordered = sorted(
        calls,
        key=lambda call: (
            rank.get(call.chrom, len(rank)),
            call.chrom,
            call.start,
            call.end,
            call.svtype
        ),
    )
    return ordered

def split_calls_by_svtype(calls: Iterable[Call]) -> dict[str, list[Call]]:
    """Split calls into a dictionary keyed by svtype."""
    svtype_groups: dict[str, list[Call]] = defaultdict(list)
    for call in calls:
        svtype_groups[call.svtype].append(call)
    return svtype_groups

def dump_calls_to_bed(
    calls: Iterable[Call], 
    dir_path: Path, 
    chrom_order: list[str] | None, 
    separate_by_sample: bool = True
) -> None:
    """Write calls to a BED file, with the pipeline's 5-column layout."""
    
    os.makedirs(dir_path, exist_ok=True)
    if separate_by_sample:
        sample_groups: dict[str, list[Call]] = defaultdict(list)
        for call in calls:
            sample_groups[call.sample_id].append(call)
        
        for sample_id in sample_groups.keys():
            sorted_calls = sort_calls(sample_groups[sample_id], chrom_order)
            with open(dir_path / f"{sample_id}.bed", "w") as f:
                for call in sorted_calls:
                    f.write(f"{call.chrom}\t{call.start}\t{call.end}\t{call.svtype}\t{'|'.join(sorted(call.sources))}\n")
        
    else:
        sorted_calls = sort_calls(calls, chrom_order)
        with open(dir_path / "merged.bed", "w") as f:
            for call in sorted_calls:
                f.write(f"{call.chrom}\t{call.start}\t{call.end}\t{call.svtype}\t{'|'.join(sorted(call.sources))}\n")
