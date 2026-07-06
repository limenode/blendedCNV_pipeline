"""Graph representation of CNV calls for consensus methodology experiments.

A sample's calls (across all callers) form an undirected overlap graph:
    node  = one CNV call
    edge  = two calls, from different callers, that reciprocally overlap

Consensus level is deliberately *not* encoded in the graph. Each consensus
methodology is a separate function `nx.Graph -> list[...]` that reads groups out
of this shared substrate (connected components, cliques, the legacy pairwise
merge, ...), so alternatives can be compared on identical inputs.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx


@dataclass(frozen=True)
class Call:
    """A single CNV call, tagged with the caller and sample it came from."""

    chrom: str
    start: int
    end: int
    svtype: str
    caller: str
    sample_id: str


def read_bed_file(path: Path) -> list[Call]:
    """Read a per-caller BED file into `Call`s.

    Expects the pipeline's 5-column layout (chrom, start, end, svtype, source),
    where `source` is the caller label. The sample id is the filename text
    before the first dot, matching the rest of the pipeline.
    """
    sample_id = path.name.split(".")[0]
    calls: list[Call] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chrom, start, end, svtype, caller = line.split("\t")[:5]
            calls.append(
                Call(chrom, int(start), int(end), svtype, caller, sample_id)
            )
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
    link_same_caller: bool = False,
    min_edge_overlap: float = 0.0,
) -> nx.Graph:
    """Build the overlap graph from CNV calls.

    Every call becomes a node (id = its index in `calls`, `call` attribute set),
    so calls with no matches still appear. Any two calls that overlap at all are
    joined by an edge whose `weight` is their `reciprocal_overlap`; no consensus
    threshold is applied here. Choosing a threshold is left to the consensus
    function, which filters edges by `weight` -- so one graph serves every
    threshold. Same-caller pairs are skipped unless `link_same_caller` is set.

    `min_edge_overlap` is only a construction floor to keep the graph sparse on
    very large cohorts; leave it at 0.0 (keep every overlap) unless edge count
    becomes a problem, and keep it well below any consensus threshold you sweep.

    Edges never cross `(sample_id, svtype, chrom)`, so calls from different
    samples stay in separate sub-graphs even when passed in together.
    """
    graph = nx.Graph()
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
            a = calls[a_id]
            for b_id in node_ids[i + 1:]:
                b = calls[b_id]
                if b.start >= a.end:
                    break  # sorted by start: no later call can overlap `a`
                if not link_same_caller and a.caller == b.caller:
                    continue
                overlap = reciprocal_overlap(a, b)
                if overlap >= min_edge_overlap:
                    graph.add_edge(a_id, b_id, weight=overlap)

    return graph


def build_sample_graphs(calls: list[Call], **kwargs) -> dict[str, nx.Graph]:
    """Build one overlap graph per sample, keyed by `sample_id`."""
    by_sample: dict[str, list[Call]] = defaultdict(list)
    for call in calls:
        by_sample[call.sample_id].append(call)
    return {
        sample_id: build_graph(sample_calls, **kwargs)
        for sample_id, sample_calls in by_sample.items()
    }

def build_sample_graphs_from_beds(
    bed_paths: Iterable[Path], **kwargs
) -> dict[str, nx.Graph]:
    """Build one overlap graph per sample, keyed by `sample_id`."""
    calls = [call for path in bed_paths for call in read_bed_file(path)]
    return build_sample_graphs(calls, **kwargs)

def build_graph_from_beds(
    bed_paths: Iterable[Path], **kwargs
) -> nx.Graph:
    """Build one overlap graph from a list of BED files."""
    calls = [call for path in bed_paths for call in read_bed_file(path)]
    return build_graph(calls, **kwargs)


@dataclass(frozen=True)
class ConsensusCall:
    """A consensus call merged from one connected component of the graph.

    `supporting_callers` is the set of distinct callers backing it (its
    consensus level = `len(supporting_callers)`); `members` are the node ids it
    was merged from, kept for provenance back to the original calls.
    """

    chrom: str
    start: int
    end: int
    svtype: str
    supporting_callers: frozenset[str]
    members: tuple[int, ...]


def consensus_by_components(
    graph: nx.Graph,
    *,
    min_nodes: int = 1,
    min_weight: float = 0.0,
) -> list[ConsensusCall]:
    """Call consensus by merging connected components of the overlap graph.

    Edges below `min_weight` are dropped first (isolated nodes are kept), then
    every remaining connected component of at least `min_nodes` calls is merged
    into one `ConsensusCall`. 
    
    With `min_nodes=1, min_weight=0.0` this returns
    every call with overlapping ones merged -- the 1-of-3 (union) set; raising
    either knob tightens toward stricter agreement.

    Note: `min_nodes` counts *nodes*, not distinct callers. Two calls from the
    same caller can land in one component via a third call, so a k-node component
    may span fewer than k callers. Filter on `len(supporting_callers)` instead if
    you want strict k-of-3 caller agreement.
    """
    filtered = nx.Graph()
    filtered.add_nodes_from(graph.nodes(data=True))
    filtered.add_edges_from(
        (u, v, data)
        for u, v, data in graph.edges(data=True)
        if data["weight"] >= min_weight
    )

    consensus: list[ConsensusCall] = []
    for component in nx.connected_components(filtered):
        if len(component) >= min_nodes:
            consensus.append(_merge_component(graph, component))
    return consensus


def _merge_component(graph: nx.Graph, component: set[int]) -> ConsensusCall:
    """Merge one component's calls into a single union-span `ConsensusCall`."""
    calls = [graph.nodes[node_id]["call"] for node_id in component]
    return ConsensusCall(
        chrom=calls[0].chrom,      # uniform within a component by construction
        start=min(call.start for call in calls),
        end=max(call.end for call in calls),
        svtype=calls[0].svtype,    # uniform within a component by construction
        supporting_callers=frozenset(call.caller for call in calls),
        members=tuple(sorted(component)),
    )
