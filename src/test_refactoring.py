# %%
from operator import attrgetter
from timeit import timeit
from pathlib import Path
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix, csgraph

chromosome_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

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
    chromosome_index: dict[str, int]

    # columnar node fields
    starts: np.ndarray
    ends: np.ndarray
    source_bits: np.ndarray
    source_names: list[str]

    # metadata columns
    chrom_idx: np.ndarray
    chrom_names: list[str]
    svtype_idx: np.ndarray
    svtype_names: list[str]
    sample_idx: np.ndarray
    sample_names: list[str]

    # overlap edges
    ov_a: np.ndarray
    ov_b: np.ndarray
    ov_key: np.ndarray

    # gap edges
    gap_a: np.ndarray
    gap_b: np.ndarray
    gap_key: np.ndarray

def retrieve_overlap(call_a: Call, call_b: Call) -> tuple[bool, float, int]:
    reciprocal_overlap = 0.0
    distance = 0

    if call_a.chrom != call_b.chrom:
        return False, 0, 0

    overlap_start = max(call_a.start, call_b.start)
    overlap_end = min(call_a.end, call_b.end)

    if overlap_start < overlap_end:
        overlap_length = overlap_end - overlap_start
        reciprocal_overlap = overlap_length / max(
            call_a.end - call_a.start, call_b.end - call_b.start
        )
        distance = 0
    else:
        distance = min(abs(call_a.start - call_b.end), abs(call_b.start - call_a.end))

    return True, reciprocal_overlap, distance

# %%
def read_bed_calls(path: str | Path) -> Iterator[Call]:
    path = Path(path) if isinstance(path, str) else path

    sample_id = path.stem

    with open(path, "r") as bed_file:
        for line in bed_file:
            if line.startswith("#"):
                continue  # Skip comment lines
            fields = line.strip().split("\t")
            if len(fields) < 5:
                continue  # Skip lines that don't have enough fields

            chrom, start, end, svtype, source = (
                fields[0],
                int(fields[1]),
                int(fields[2]),
                fields[3],
                fields[4],
            )
            yield Call(
                chrom=chrom, start=start, end=end, svtype=svtype, source=source, sample_id=sample_id
            )

def sort_into_genome_order(
    calls: Iterable[Call], chromosome_order: Iterable[str]
) -> list[Call]:
    """Put calls into canonical order: genome chromosome order, then (start, end)."""
    by_chrom = defaultdict(list)
    for call in calls:
        by_chrom[call.chrom].append(call)

    # Every field is in the key, not just (start, end): ~57% of calls share an
    # exact (chrom, start, end) with their neighbour, and list.sort is stable, so
    # a shorter key would leave those ties resolved by input order.
    key = attrgetter("start", "end", "svtype", "source", "sample_id")
    ordered: list[Call] = []
    for chrom in chromosome_order:
        group = by_chrom.pop(chrom, None)
        if group:
            group.sort(key=key)
            ordered += group

    for chrom in sorted(by_chrom):  # contigs absent from the genome file
        by_chrom[chrom].sort(key=key)
        ordered += by_chrom[chrom]

    return ordered


def build_callset(
    calls: Iterable[Call], *, chromosome_order: Iterable[str] = chromosome_order) -> CallSet:
    calls_list = sort_into_genome_order(calls, chromosome_order)

    ov_a: list[int] = []
    ov_b: list[int] = []
    ov_key: list[float] = []
    gap_a: list[int] = []
    gap_b: list[int] = []
    gap_key: list[int] = []

    starts: list[int] = []
    ends: list[int] = []
    bits: list[int] = []
    source_names: list[str] = []
    bit_of_source: dict[str, int] = {}

    chrom_ids: list[int] = []
    chrom_names: list[str] = []
    svtype_ids: list[int] = []
    svtype_names: list[str] = []
    index_of_svtype: dict[str, int] = {}
    sample_ids: list[int] = []
    sample_names: list[str] = []
    index_of_sample: dict[str, int] = {}

    chromosome_index: dict[str, int] = {}
    svtype_connected_component = defaultdict(list)
    previous_chrom = None

    for current_call_index, current_call in enumerate(calls_list):
        chrom = current_call.chrom
        start = current_call.start
        end = current_call.end
        svtype = current_call.svtype

        if chrom != previous_chrom:
            # chromosomes are contiguous after sorting, so each is seen exactly once and is always a new name
            chromosome_index[chrom] = current_call_index
            chrom_names.append(chrom)
            svtype_connected_component.clear()
            previous_chrom = chrom

        starts.append(start)
        ends.append(end)
        bit = bit_of_source.get(current_call.source)
        if bit is None:
            bit = bit_of_source[current_call.source] = 1 << len(source_names)
            source_names.append(current_call.source)
        bits.append(bit)

        chrom_ids.append(len(chrom_names) - 1)

        svtype_id = index_of_svtype.get(svtype)
        if svtype_id is None:
            svtype_id = index_of_svtype[svtype] = len(svtype_names)
            svtype_names.append(svtype)
        svtype_ids.append(svtype_id)

        sample_id = index_of_sample.get(current_call.sample_id)
        if sample_id is None:
            sample_id = index_of_sample[current_call.sample_id] = len(sample_names)
            sample_names.append(current_call.sample_id)
        sample_ids.append(sample_id)

        current_size = end - start
        create_new_connected_component = True

        # sample_id must be in the key: without it, edges join calls from
        # different samples and components merge unrelated genomes
        key = (current_call.sample_id, chrom, svtype)
        component_of_interest = svtype_connected_component[key]

        for i in component_of_interest:
            prev_call = calls_list[i]
            prev_end = prev_call.end
            overlap_end = min(prev_end, end)

            if start < overlap_end:
                prev_size = prev_end - prev_call.start
                ov_a.append(i)
                ov_b.append(current_call_index)
                ov_key.append((overlap_end - start) / max(prev_size, current_size))
                create_new_connected_component = False
            else:
                distance = (
                    start - prev_end
                )  # start <= prev_call.end due to sorting, so this is always positive
                gap_a.append(i)
                gap_b.append(current_call_index)
                gap_key.append(distance)
                if distance == 0:
                    create_new_connected_component = False

        if create_new_connected_component:
            svtype_connected_component[key] = [current_call_index]  # Start a new connected component for this svtype
        else:
            component_of_interest.append(current_call_index)  # Add to the existing connected component

    n = len(calls_list)
    n_ov = len(ov_a)
    n_gap = len(gap_a)

    # Sort each edge kind by its own key
    ov_key_arr = np.fromiter(ov_key, np.float64, n_ov)
    ov_order = np.argsort(ov_key_arr)
    gap_key_arr = np.fromiter(gap_key, np.int64, n_gap)
    gap_order = np.argsort(gap_key_arr)

    return CallSet(
        calls=calls_list,
        chromosome_index=chromosome_index,
        starts=np.fromiter(starts, np.int64, n),
        ends=np.fromiter(ends, np.int64, n),
        source_bits=np.fromiter(bits, np.int64, n),
        source_names=source_names,
        chrom_idx=np.fromiter(chrom_ids, np.int32, n),
        chrom_names=chrom_names,
        svtype_idx=np.fromiter(svtype_ids, np.int32, n),
        svtype_names=svtype_names,
        sample_idx=np.fromiter(sample_ids, np.int32, n),
        sample_names=sample_names,
        ov_a=np.fromiter(ov_a, np.int64, n_ov)[ov_order],
        ov_b=np.fromiter(ov_b, np.int64, n_ov)[ov_order],
        ov_key=ov_key_arr[ov_order],
        gap_a=np.fromiter(gap_a, np.int64, n_gap)[gap_order],
        gap_b=np.fromiter(gap_b, np.int64, n_gap)[gap_order],
        gap_key=gap_key_arr[gap_order],
    )

# %%
CallSource = CallSet | Iterable[Call]

def collect_callsets(
    sources: Iterable[CallSource], *, chromosome_order: Iterable[str] = chromosome_order
) -> CallSet:
    """Pool calls from any mix of CallSets and raw Call iterables into one CallSet."""
    calls: list[Call] = []
    for source in sources:
        calls.extend(source.calls if isinstance(source, CallSet) else source)
    return build_callset(calls, chromosome_order=chromosome_order)

# %%
@dataclass(frozen=True, slots=True)
class EdgeSelection:
    a: np.ndarray
    b: np.ndarray
    min_reciprocal_overlap: float
    max_padding: int | None

    def __len__(self) -> int:
        return len(self.a)

# %%
def filter_edges(
    callset: CallSet,
    min_reciprocal_overlap: float = 0.0,
    max_padding: int | None = None,
    *,
    allow_mixed: bool = False,
) -> EdgeSelection:
    """Select edges passing reciprocal overlap and/or padding thresholds."""

    if min_reciprocal_overlap > 0.0 and max_padding is not None and not allow_mixed:
        raise ValueError(
            f"min_reciprocal_overlap={min_reciprocal_overlap} and max_padding={max_padding} are both set, \
            but allow_mixed is False. This drops overlapping pairs below the threshold while keeping \
            non-overlapping pairs within the padding, which is not interpretable."
        )

    ov_key = callset.ov_key

    recip_idx = np.searchsorted(ov_key, min_reciprocal_overlap, side="left")

    if max_padding is None:
        return EdgeSelection(
            a=callset.ov_a[recip_idx:],
            b=callset.ov_b[recip_idx:],
            min_reciprocal_overlap=min_reciprocal_overlap,
            max_padding=None,
        )

    padding_idx = np.searchsorted(callset.gap_key, max_padding, side="right")

    if padding_idx == 0:
        a, b = callset.ov_a[recip_idx:], callset.ov_b[recip_idx:]
    elif recip_idx == len(ov_key):
        a, b = callset.gap_a[:padding_idx], callset.gap_b[:padding_idx]
    else:
        a = np.concatenate((callset.ov_a[recip_idx:], callset.gap_a[:padding_idx]))
        b = np.concatenate((callset.ov_b[recip_idx:], callset.gap_b[:padding_idx]))

    return EdgeSelection(
        a=a, b=b, min_reciprocal_overlap=min_reciprocal_overlap, max_padding=max_padding
    )


# %%
@dataclass(frozen=True, slots=True)
class MergedCallSet:
    representative: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    source_bits: np.ndarray
    n_calls: np.ndarray
    component_id: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.starts)

    @property
    def n_sources(self) -> np.ndarray:
        return np.bitwise_count(self.source_bits) # Count the number of unique sources for each merged call

def merge_components(
    callset: CallSet,
    selection: EdgeSelection | None = None,
    *,
    min_reciprocal_overlap: float = 0.0,
    max_padding: int | None = None,
    min_calls: int = 1,
    min_sources: int = 1
) -> MergedCallSet:
    if selection is None:
        selection = filter_edges(
            callset,
            min_reciprocal_overlap=min_reciprocal_overlap,
            max_padding=max_padding,
        )

    n = len(callset.calls)
    graph = coo_matrix((np.ones(len(selection), np.int8), (selection.a, selection.b)), shape=(n, n))

    n_components, labels = csgraph.connected_components(graph, directed=False, return_labels=True)

    # One reduction per column.
    starts = np.full(n_components, np.iinfo(np.int64).max, dtype=np.int64)
    ends = np.zeros(n_components, dtype=np.int64)
    source_bits = np.zeros(n_components, dtype=np.int64)
    representative = np.full(n_components, n, dtype=np.int64)

    np.minimum.at(starts, labels, callset.starts)
    np.maximum.at(ends, labels, callset.ends)
    np.bitwise_or.at(source_bits, labels, callset.source_bits)
    np.minimum.at(representative, labels, np.arange(n, dtype=np.int64))
    n_calls = np.bincount(labels, minlength=n_components)

    component_id = np.arange(n_components, dtype=np.int64)
    if min_calls > 1 or min_sources > 1:
        keep = ((n_calls >= min_calls) & (np.bitwise_count(source_bits) >= min_sources))
        representative, starts, ends, source_bits, n_calls, component_id = (
            x[keep] for x in (representative, starts, ends, source_bits, n_calls, component_id)
        )

    return MergedCallSet(
        representative=representative,
        starts=starts,
        ends=ends,
        source_bits=source_bits,
        n_calls=n_calls,
        component_id=component_id,
        labels=labels,
    )


# %%
def source_strings_for(source_names: list[str], masks: list[int]) -> list[str]:
    """Render each source bitmask as a pipe-joined name string."""
    n_sources = len(source_names)

    if n_sources <= 16:
        table = [
            "|".join(sorted(source_names[i] for i in range(n_sources) if mask >> i & 1))
            for mask in range(1 << n_sources)
        ]
        return [table[mask] for mask in masks]

    cache: dict[int, str] = {}
    rendered: list[str] = []
    for mask in masks:
        text = cache.get(mask)
        if text is None:
            text = cache[mask] = "|".join(sorted(
                name for i, name in enumerate(source_names) if mask >> i & 1
            ))
        rendered.append(text)
    return rendered


def write_merged_bed(
    callset: CallSet,
    merged: MergedCallSet,
    path: str | Path,
    *,
    include_sample: bool = False,
) -> int:
    """Write a merged call set to BED, returning the number of rows written."""
    representative = merged.representative
    chrom_ids = callset.chrom_idx[representative]
    svtype_ids = callset.svtype_idx[representative]
    sample_ids = callset.sample_idx[representative]

    sample_rank = np.argsort(np.argsort(callset.sample_names))[sample_ids]
    svtype_rank = np.argsort(np.argsort(callset.svtype_names))[svtype_ids]
    order = np.lexsort((sample_rank, svtype_rank, merged.ends, merged.starts, chrom_ids))

    chrom_ids = chrom_ids[order].tolist()
    svtype_ids = svtype_ids[order].tolist()
    starts = merged.starts[order].tolist()
    ends = merged.ends[order].tolist()
    sources = source_strings_for(callset.source_names, merged.source_bits[order].tolist())

    chrom_names = callset.chrom_names
    svtype_names = callset.svtype_names

    if include_sample:
        sample_names = callset.sample_names
        sample_ids = sample_ids[order].tolist()
        rows = [
            f"{chrom_names[c]}\t{s}\t{e}\t{svtype_names[v]}\t{src}\t{sample_names[p]}\n"
            for c, s, e, v, src, p in zip(
                chrom_ids, starts, ends, svtype_ids, sources, sample_ids
            )
        ]
    else:
        rows = [
            f"{chrom_names[c]}\t{s}\t{e}\t{svtype_names[v]}\t{src}\n"
            for c, s, e, v, src in zip(chrom_ids, starts, ends, svtype_ids, sources)
        ]

    with open(path, "w") as bed_file:
        bed_file.writelines(rows)
    return len(rows)

# %%
cs1 = build_callset(
    read_bed_calls(
        "/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/1000G/HG00096.bed"
    )
)
cs2 = build_callset(
    read_bed_calls(
        "/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/HGSVC3/HG00096.bed"
    )
)
cs3 = build_callset(
    read_bed_calls(
        "/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/ont_vienna/HG00096.bed"
    )
)
print(len(cs1.calls), "calls in callset 1; ", len(cs1.ov_key), "overlap edges.")
print(len(cs2.calls), "calls in callset 2; ", len(cs2.ov_key), "overlap edges.")
print(len(cs3.calls), "calls in callset 3; ", len(cs3.ov_key), "overlap edges.")
print(
    len(cs1.calls) + len(cs2.calls) + len(cs3.calls),
    "calls in total; ",
    len(cs1.ov_key) + len(cs2.ov_key) + len(cs3.ov_key),
    "overlap edges.",
)
timeit(lambda: collect_callsets([cs1, cs2, cs3]), number=1)

merged_callset = collect_callsets([cs1, cs2, cs3])
print(len(merged_callset.calls), "calls in merged callset.")
print(len(merged_callset.ov_key), "edges in merged callset.")

# %%
merged_callset = collect_callsets([cs1, cs2, cs3])
print(timeit(
    lambda: merge_components(
        merged_callset, max_padding=0, min_calls=1, min_sources=1
    ),
    number=10,
))
merged_set = merge_components(
    merged_callset, max_padding=0, min_calls=1, min_sources=1
)
print(len(merged_set.starts))
# %%
test_path = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/src/test")
print(timeit(
    lambda: write_merged_bed(merged_callset, merged_set, test_path / "output.bed"),
    number=1,
))


# %%
import glob

list_of_beds = glob.glob("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/benchmark/*/*.bed")
all_bm_callset = collect_callsets([read_bed_calls(bed) for bed in list_of_beds])


# %%
timeit(
    lambda: write_merged_bed(all_bm_callset, merge_components(all_bm_callset), test_path / "merged_all_benchmarks.bed", include_sample=True),
    number=1
)
