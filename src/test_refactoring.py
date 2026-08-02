# %%
from timeit import timeit
from pathlib import Path
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
def read_bed_file_into_callset(bed_file_path: str | Path) -> CallSet:
    if isinstance(bed_file_path, str):
        bed_file_path = Path(bed_file_path)

    calls: list[Call] = []
    edges: list[tuple[int, int, float, int]] = []
    chromosome_index: dict[str, int] = {}

    svtype_connected_component = defaultdict(list)

    with open(bed_file_path, 'r') as bed_file:
        sample_id = bed_file_path.stem
        for line in bed_file:
            if line.startswith('#'):
                continue  # Skip comment lines
            fields = line.strip().split('\t')
            if len(fields) < 5:
                continue  # Skip lines that don't have enough fields

            chrom, start, end, svtype, source = fields[0], int(fields[1]), int(fields[2]), fields[3], fields[4]
            current_call = Call(chrom=chrom, start=start, end=end, svtype=svtype, source=source, sample_id=sample_id)
            calls.append(current_call)
            current_call_index = len(calls) - 1

            if chrom not in chromosome_index:
                chromosome_index[chrom] = current_call_index  # Store the index of the first call for this chromosome
                svtype_connected_component.clear()

            # Create edges with all previous calls in the same connected component
            create_new_connected_component = True
            component_of_interest = svtype_connected_component[svtype]

            for i in component_of_interest:
                # inline retrieve overlap logic
                prev_call: Call = calls[i]
                overlap_end = min(prev_call.end, end)

                if start < overlap_end:
                    prev_size = prev_call.end - prev_call.start
                    curr_size = end - start
                    reciprocal_overlap = (overlap_end - start) / max(prev_size, curr_size)
                    distance = 0
                    create_new_connected_component = False
                else:
                    reciprocal_overlap = 0
                    distance = start - prev_call.end # start <= prev_call.end due to sorting, so this is always positive
                    if distance == 0:
                        create_new_connected_component = False

                edges.append((i, current_call_index, reciprocal_overlap, distance))

            if create_new_connected_component:
                svtype_connected_component[svtype] = [current_call_index]  # Start a new connected component for this svtype
            else:
                component_of_interest.append(current_call_index)  # Add to the existing connected component

    # print(len(calls), "calls and", len(edges), "edges created in the call set.")
    return CallSet(calls=calls, edges=edges, chromosome_index=chromosome_index)

# %%
print(timeit(lambda: read_bed_file_into_graph("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/1000G/HG00096.bed"), number=10))
print(timeit(lambda: read_bed_file_into_callset("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/1000G/HG00096.bed"), number=10))

# %%
chromosome_order = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10",
                    "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
                    "chr20", "chr21", "chr22", "chrX", "chrY"]

# %%
def merge_callsets(list_of_callsets: list[CallSet]) -> CallSet:
    rank = {chrom: i for i, chrom in enumerate(chromosome_order or [])}

    decorated = [
        (set_idx, local_idx, call)
        for set_idx, callset in enumerate(list_of_callsets)
        for local_idx, call in enumerate(callset.calls)
    ]

    merged = (
        sorted(decorated, key=lambda item: (
            rank.get(item[2].chrom, len(rank)),
            item[2].chrom,
            item[2].start,
            item[2].end)
        )
    )

    remap: list[list[int]] = [[-1] * len(cs.calls) for cs in list_of_callsets]

    calls: list[Call] = []
    origin: list[int] = []
    chromosome_index: dict[str, int] = {}

    for new_idx, (set_idx, local_idx, call) in enumerate(merged):
        calls.append(call)
        origin.append(set_idx)
        remap[set_idx][local_idx] = new_idx

        if call.chrom not in chromosome_index:
            chromosome_index[call.chrom] = new_idx

    edges: list[tuple[int, int, dict[str, float | int]]] = [
        (remap[set_idx][u], remap[set_idx][v], data)
        for set_idx, callset in enumerate(list_of_callsets)
        for u, v, data in callset.edges
    ]

    #

    return CallSet(calls=calls, edges=edges, chromosome_index=chromosome_index)

# %%
cs1 = read_bed_file_into_callset("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/1000G/HG00096.bed")
cs2 = read_bed_file_into_callset("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/HGSVC3/HG00096.bed")
print(len(cs1.calls), "calls in callset 1; ", len(cs1.edges), "edges.")
print(len(cs2.calls), "calls in callset 2; ", len(cs2.edges), "edges.")
print(len(cs1.calls) + len(cs2.calls), "calls in total; ", len(cs1.edges) + len(cs2.edges), "edges.")
# timeit(lambda: merge_callsets([cs1, cs2]), number=1)

# %%
merged_callset = merge_callsets([cs1, cs2])
print(len(merged_callset.calls), "calls in merged callset.")
print(len(merged_callset.edges), "edges in merged callset.")
