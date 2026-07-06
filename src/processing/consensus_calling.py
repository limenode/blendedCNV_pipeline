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
    reciprocal_threshold: float = 0.5,
    link_same_caller: bool = False,
) -> nx.Graph:
    """Build the overlap graph from CNV calls.

    Every call becomes a node (id = its index in `calls`, `call` attribute set),
    so calls with no matches still appear. 
    An edge joins two calls when their `reciprocal_overlap` meets 
    `reciprocal_threshold`; the fraction is stored as the edge `weight`. 
    Same-caller pairs are skipped unless `link_same_caller` is set.

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
                if overlap >= reciprocal_threshold:
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
