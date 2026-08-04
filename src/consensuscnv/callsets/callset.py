"""The CallSet contains interval calls in columnar form, plus a threshold-independent
overlap graph over them.

`build_callset` runs one sweep over genome-sorted calls and records every pair
that could ever be joined, as two edge lists:

- **overlap edges** carry a reciprocal-overlap key in ``(0, 1]``
- **gap edges** carry a base-pair distance ``>= 0``

The two kinds are mutually exclusive, and each is stored sorted by its own key.
No threshold is applied at build time -- one CallSet serves every parameter
point, and `filter_edges` selects a contiguous slice per threshold.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from operator import attrgetter

import numpy as np

from consensuscnv.callsets.calls import Call

DEFAULT_CHROMOSOME_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


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

    def __len__(self) -> int:
        return len(self.calls)


def sort_into_genome_order(
    calls: Iterable[Call], chromosome_order: Iterable[str]
) -> list[Call]:
    """Put calls into canonical order: genome chromosome order, then (start, end).

    A plain sort on (chrom, start, end) would order chromosomes lexicographically
    -- chr1, chr10, chr11, chr2 -- and that ordering becomes `chrom_names` and so
    reaches the BED output, hence the explicit chromosome order.
    """
    by_chrom = defaultdict(list)
    for call in calls:
        by_chrom[call.chrom].append(call)

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
    calls: Iterable[Call],
    *,
    chromosome_order: Iterable[str] = DEFAULT_CHROMOSOME_ORDER,
) -> CallSet:
    """Build a CallSet from an interable of calls."""
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
        source = current_call.source
        sample_id = current_call.sample_id

        if chrom != previous_chrom:
            # chromosomes are contiguous after sorting, so each is seen exactly once and is always a new name
            chromosome_index[chrom] = current_call_index
            chrom_names.append(chrom)
            svtype_connected_component.clear()
            previous_chrom = chrom

        starts.append(start)
        ends.append(end)

        chrom_ids.append(len(chrom_names) - 1)

        bit = bit_of_source.get(source)
        if bit is None:
            bit = bit_of_source[source] = 1 << len(source_names)
            source_names.append(source)
        bits.append(bit)

        svtype_index = index_of_svtype.get(svtype)
        if svtype_index is None:
            svtype_index = index_of_svtype[svtype] = len(svtype_names)
            svtype_names.append(svtype)
        svtype_ids.append(svtype_index)

        sample_index = index_of_sample.get(sample_id)
        if sample_index is None:
            sample_index = index_of_sample[sample_id] = len(sample_names)
            sample_names.append(sample_id)
        sample_ids.append(sample_index)

        current_size = end - start
        create_new_connected_component = True

        key = (sample_id, chrom, svtype)
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


CallSource = CallSet | Iterable[Call]


def collect_callsets(
    sources: Iterable[CallSource],
    *,
    chromosome_order: Iterable[str] = DEFAULT_CHROMOSOME_ORDER,
) -> CallSet:
    """Pool calls from any mix of CallSets and raw Call iterables into one CallSet."""
    calls: list[Call] = []
    for source in sources:
        calls.extend(source.calls if isinstance(source, CallSet) else source)
    return build_callset(calls, chromosome_order=chromosome_order)
