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
import glob
from pathlib import Path
from typing import Iterable
import networkx as nx

from utils import PipelineConfig


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
    
    def bed_str(self) -> str:
        return f"{self.chrom}\t{self.start}\t{self.end}\t{self.svtype}\t{'|'.join(sorted(self.supporting_callers))}"


def consensus_by_components(
    graph: nx.Graph,
    *,
    min_nodes: int = 1,
    min_weight: float = 0.0,
    chrom_order: list[str] | None = None,
) -> list[ConsensusCall]:
    """Call consensus by merging connected components of the overlap graph.

    Edges below `min_weight` are dropped first (isolated nodes are kept), then
    every remaining connected component of at least `min_nodes` calls is merged
    into one `ConsensusCall`.

    `min_nodes=1, min_weight=0.0` returns all calls merged (the 1-of-3 (union) set).

    Output is sorted by genomic coordinate. `chrom_order` sets the chromosome
    ordering; contigs not in it sort after the known ones. When omitted, chroms
    sort lexicographically (`chr1, chr10, chr11, ... chr2`).
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

    # Sort by genomic coordinate; unknown contigs sort after the known ones.
    rank = {chrom: i for i, chrom in enumerate(chrom_order or [])}
    consensus.sort(key=lambda c: (rank.get(c.chrom, len(rank)), c.chrom, c.start, c.end))
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

def compute_consensus_from_beds(config: PipelineConfig, weight_threshold: float = 0.5):
    """Compute consensus calls from per-caller BED files and write them to the layout.

    Args:
        config (PipelineConfig): The pipeline configuration.
        weight_threshold (float, optional): The minimum weight for edges in the overlap graph. Defaults to 0.5.
    """
    
    layout = config.layout
    input_keys = config.input.keys()
    
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
                consensus_calls = consensus_by_components(
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
                