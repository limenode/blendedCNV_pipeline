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
from consensuscnv.callsets.registry import (
    CHROMOSOMES,
    DEFAULT_CHROMOSOME_ORDER,
    SAMPLES,
    SOURCES,
    SVTYPES,
)


@dataclass
class CallSet:
    """
    A set of calls in columnar form, plus a threshold-independent overlap graph.

    Notes:
        - To get all chromosomes present in the CallSet, use `np.unique(cs.chrom_idx)`.
    """
    calls: list[Call]

    # columnar node fields
    starts: np.ndarray
    ends: np.ndarray

    # metadata columns
    chrom_idx: np.ndarray
    svtype_idx: np.ndarray
    sample_idx: np.ndarray
    source_bits: np.ndarray

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
    """Put calls into canonical order: (chrom, start, end, svtype, source, sample_id)."""
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
    chrom_ids: list[int] = []
    svtype_ids: list[int] = []
    sample_ids: list[int] = []

    bit_of_source: dict[str, int] = {}
    id_of_svtype: dict[str, int] = {}
    id_of_sample: dict[str, int] = {}

    cached_source_bit = bit_of_source.get
    cached_svtype_id = id_of_svtype.get
    cached_sample_id = id_of_sample.get

    svtype_connected_component = defaultdict(list)
    previous_chrom = None
    chrom_id = -1

    for current_call_index, current_call in enumerate(calls_list):
        chrom = current_call.chrom
        start = current_call.start
        end = current_call.end
        svtype = current_call.svtype
        source = current_call.source
        sample_id = current_call.sample_id

        if chrom != previous_chrom:
            # chromosomes are contiguous after sorting, so each is seen exactly once and is always a new name
            chrom_id = CHROMOSOMES.intern(chrom)
            svtype_connected_component.clear()
            previous_chrom = chrom

        starts.append(start)
        ends.append(end)
        chrom_ids.append(chrom_id)

        bit = cached_source_bit(source)
        if bit is None:
            bit = bit_of_source[source] = 1 << SOURCES.intern(source)
        bits.append(bit)

        svtype_index = cached_svtype_id(svtype)
        if svtype_index is None:
            svtype_index = id_of_svtype[svtype] = SVTYPES.intern(svtype)
        svtype_ids.append(svtype_index)

        sample_index = cached_sample_id(sample_id)
        if sample_index is None:
            sample_index = id_of_sample[sample_id] = SAMPLES.intern(sample_id)
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
        starts=np.fromiter(starts, np.int64, n),
        ends=np.fromiter(ends, np.int64, n),
        source_bits=np.fromiter(bits, np.int64, n),
        chrom_idx=np.fromiter(chrom_ids, np.int32, n),
        svtype_idx=np.fromiter(svtype_ids, np.int32, n),
        sample_idx=np.fromiter(sample_ids, np.int32, n),
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
