# %%
from operator import attrgetter
from timeit import timeit
from pathlib import Path
from collections.abc import Iterable, Iterator
import networkx as nx
from dataclasses import dataclass
from collections import defaultdict

# %%
@dataclass(frozen=True, slots=True)
class Call:
    chrom: str
    start: int
    end: int
    svtype: str
    source: str
    sample_id: str

@dataclass
class CallSet:
    calls: list[Call]
    edges: list[tuple[int, int, float, int]] # (node_a_index, node_b_index, reciprocal_overlap, distance: int)
    chromosome_index: dict[str, int] # Maps chromosome names to their index in the calls list for quick access


def retrieve_overlap(call_a: Call, call_b: Call) -> tuple[bool, float, int]:
    reciprocal_overlap = 0.0
    distance = 0

    if call_a.chrom != call_b.chrom:
        return False, 0, 0

    overlap_start = max(call_a.start, call_b.start)
    overlap_end = min(call_a.end, call_b.end)

    if overlap_start < overlap_end:
        overlap_length = overlap_end - overlap_start
        reciprocal_overlap = overlap_length / max(call_a.end - call_a.start, call_b.end - call_b.start)
        distance = 0
    else:
        distance = min(abs(call_a.start - call_b.end), abs(call_b.start - call_a.end))

    return True, reciprocal_overlap, distance

# %%
def read_bed_file_into_graph(bed_file_path: str | Path) -> nx.Graph:
    if isinstance(bed_file_path, str):
        bed_file_path = Path(bed_file_path)

    # Read the BED file and create a graph
    G = nx.Graph()
    counter = 0

    with open(bed_file_path, 'r') as bed_file:
        sample_id = bed_file_path.stem
        # print(f"Reading BED file for sample: {sample_id}")
        for line in bed_file:
            if line.startswith('#'):
                continue  # Skip comment lines
            fields = line.strip().split('\t')
            if len(fields) < 3:
                continue  # Skip lines that don't have enough fields

            chrom, start, end, svtype, source = fields[0], int(fields[1]), int(fields[2]), fields[3], fields[4]
            call = Call(chrom=chrom, start=start, end=end, svtype=svtype, source=source, sample_id=sample_id)

            G.add_node(counter, call=call)

            if counter == 0:
                counter += 1
                continue    # Early exit for the first call

            # Add edge to previous call
            prev_call = G.nodes[counter - 1]['call']
            on_same_chrom, reciprocal_overlap, distance = retrieve_overlap(prev_call, call)
            if on_same_chrom:
                G.add_edge(counter - 1, counter, reciprocal_overlap=reciprocal_overlap, distance=distance)

            counter += 1

    # print(len(G.nodes), "nodes and", len(G.edges), "edges created in the graph.")
    # print(f"counter: {counter}")

    return G

# %%
def iterate_bed(path: str | Path) -> Iterator[Call]:
    path = Path(path) if isinstance(path, str) else path

    sample_id = path.stem

    with open(path, 'r') as bed_file:
        for line in bed_file:
            if line.startswith('#'):
                continue  # Skip comment lines
            fields = line.strip().split('\t')
            if len(fields) < 5:
                continue  # Skip lines that don't have enough fields

            chrom, start, end, svtype, source = fields[0], int(fields[1]), int(fields[2]), fields[3], fields[4]
            yield Call(chrom=chrom, start=start, end=end, svtype=svtype, source=source, sample_id=sample_id)

def build_callset(calls: Iterable[Call]) -> CallSet:
    calls_list = list(calls)
    edges: list[tuple[int, int, float, int]] = []
    chromosome_index: dict[str, int] = {}

    svtype_connected_component = defaultdict(list)

    previous_chrom = None
    previous_start = -1

    for current_call_index, current_call in enumerate(calls_list):
        chrom = current_call.chrom
        start = current_call.start
        end = current_call.end
        svtype = current_call.svtype

        if chrom != previous_chrom:
            if chrom in chromosome_index:
                raise ValueError(f"Chromosome {chrom} appears multiple times in the input calls. Ensure that calls are sorted by chromosome and start position.")

            chromosome_index[chrom] = current_call_index
            svtype_connected_component.clear()
            previous_chrom = chrom
        elif start < previous_start:
            raise ValueError(f"Calls are not sorted by start position within chromosome {chrom}. Ensure that calls are sorted by chromosome and start position.")
        previous_start = start

        current_size = end - start
        create_new_connected_component = True
        component_of_interest = svtype_connected_component[svtype]

        for i in component_of_interest:
            prev_call = calls_list[i]
            prev_end = prev_call.end
            overlap_end = min(prev_end, end)

            if start < overlap_end:
                prev_size = prev_end - prev_call.start
                reciprocal_overlap = (overlap_end - start) / max(prev_size, current_size)
                distance = 0
                create_new_connected_component = False
            else:
                reciprocal_overlap = 0
                distance = start - prev_end # start <= prev_call.end due to sorting, so this is always positive
                if distance == 0:
                    create_new_connected_component = False

            edges.append((i, current_call_index, reciprocal_overlap, distance))

        if create_new_connected_component:
            svtype_connected_component[svtype] = [current_call_index]  # Start a new connected component for this svtype
        else:
            component_of_interest.append(current_call_index)  # Add to the existing connected component

    return CallSet(calls=calls_list, edges=edges, chromosome_index=chromosome_index)

# %%
print(timeit(lambda: read_bed_file_into_graph("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/1000G/HG00096.bed"), number=10))
print(timeit(lambda: build_callset(iterate_bed("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/1000G/HG00096.bed")), number=10))

# %%
chromosome_order = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10",
                    "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
                    "chr20", "chr21", "chr22", "chrX", "chrY"]

# %%
def merge_callsets(list_of_callsets: list[CallSet]) -> CallSet:
    by_chrom = defaultdict(list)
    for callset in list_of_callsets:
        for call in callset.calls:
            by_chrom[call.chrom].append(call)

    key = attrgetter("start", "end")
    merged: list[Call] = []
    for chrom in chromosome_order:
        group = by_chrom.pop(chrom, None)
        if group:
            group.sort(key=key)
            merged += group

    for chrom in sorted(by_chrom):
        by_chrom[chrom].sort(key=key)
        merged += by_chrom[chrom]

    return build_callset(merged)

# %%
cs1 = build_callset(iterate_bed("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/1000G/HG00096.bed"))
cs2 = build_callset(iterate_bed("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/HGSVC3/HG00096.bed"))
cs3 = build_callset(iterate_bed("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/ont_vienna/HG00096.bed"))
print(len(cs1.calls), "calls in callset 1; ", len(cs1.edges), "edges.")
print(len(cs2.calls), "calls in callset 2; ", len(cs2.edges), "edges.")
print(len(cs3.calls), "calls in callset 3; ", len(cs3.edges), "edges.")
print(len(cs1.calls) + len(cs2.calls) + len(cs3.calls), "calls in total; ", len(cs1.edges) + len(cs2.edges) + len(cs3.edges), "edges.")
timeit(lambda: merge_callsets([cs1, cs2, cs3]), number=1)

# %%
merged_callset = merge_callsets([cs1, cs2, cs3])
print(len(merged_callset.calls), "calls in merged callset.")
print(len(merged_callset.edges), "edges in merged callset.")
