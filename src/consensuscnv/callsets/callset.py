"""The CallSet contains interval calls in columnar form, plus a threshold-independent
overlap graph over them.

`build_callset` runs one sweep over genome-sorted calls and records every pair
that could ever be joined, as two edge lists:

- **overlap edges** carry a reciprocal-overlap key in ``(0, 1]``
- **gap edges** carry a base-pair distance in ``[0, search_radius]``

The two kinds are mutually exclusive, and each is stored sorted by its own key.
No threshold is applied at build time -- one CallSet serves every parameter
point, and `filter_edges` selects a contiguous slice per threshold.

Two build-time choices bound what the graphs compute edges for:

- `partition_by` names the Call fields an edge never crosses. The default,
  ``("svtype", "sample_id")``, is what consensus calling needs; dropping
  ``sample_id`` gives a cross-sample graph over a cohort.
- `search_radius` is the widest gap a recorded edge spans, and so the widest
  `max_padding` that `filter_edges` can serve. Overlap edges are always complete.
"""

import glob
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path

import numpy as np

from consensuscnv.callsets.bed_io import read_bed_calls
from consensuscnv.callsets.calls import PARTITION_FIELDS, Call, normalize_partition
from consensuscnv.callsets.registry import (
    CHROMOSOMES,
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

    # what the edge lists were built under; see the module docstring
    partition_by: tuple[str, ...] = PARTITION_FIELDS
    search_radius: int = 0

    def __len__(self) -> int:
        return len(self.calls)

    def require_partition(self, *fields: str) -> None:
        """Raise unless every one of `fields` partitions this graph.

        A component is uniform in a field only if no edge crosses it, so anything
        that reads a per-component chrom / svtype / sample off one member call
        has to check here first.
        """
        missing = [field for field in fields if field not in self.partition_by]
        if missing:
            raise ValueError(
                f"components of this CallSet may span {missing}: it was built with "
                f"partition_by={self.partition_by}, so no per-component value of "
                f"{missing} exists"
            )


def sort_into_genome_order(calls: Iterable[Call], chromosome_order: Iterable[str]) -> list[Call]:
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


def _resolve_chromosome_order(chromosome_order: Iterable[str] | None) -> tuple[str, ...]:
    """The order to sort chromosomes into, defaulting to the registry's own.

    `None` means "whatever `seed_chromosomes` put in the registry", which is the
    genome file's order and the same order `write_merged_bed` will sort ids into.
    """
    if chromosome_order is not None:
        return tuple(chromosome_order)
    if not CHROMOSOMES.names:
        raise ValueError(
            "The chromosome registry is empty, so there is no genome order to sort "
            "into. Call callsets.seed_chromosomes(...) with the chromosomes of the "
            "genome being analysed -- read_genome_file(genome_file) returns "
            "them -- before building a CallSet, or pass chromosome_order explicitly."
        )
    return tuple(CHROMOSOMES.names)


def build_callset(
    calls: Iterable[Call],
    *,
    chromosome_order: Iterable[str] | None = None,
    partition_by: Iterable[str] = PARTITION_FIELDS,
    search_radius: int = 0,
) -> CallSet:
    """Build a CallSet from an iterable of calls.

    `chromosome_order` defaults to the chromosome registry's order, which is what
    `seed_chromosomes` put there.

    `partition_by` names the Call fields an edge never crosses;
    chromosome always partitions. `search_radius` is the widest gap, in base
    pairs, a recorded edge spans -- and therefore the widest `max_padding`
    the CallSet can later be filtered at. Both are recorded on the result.
    """
    partition_by = normalize_partition(partition_by)
    if search_radius < 0:
        raise ValueError(f"search_radius={search_radius}; must be >= 0")
    key_of = attrgetter(*partition_by) if partition_by else (lambda call: None)

    calls_list = sort_into_genome_order(calls, _resolve_chromosome_order(chromosome_order))

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

    # Per partition, the calls a later call could still form an edge with: those
    # whose end is within `search_radius` of the current start. Calls arrive in
    # start order, so once a member falls behind that cutoff no later call can
    # reach it and it is dropped. Every pair within the radius is compared
    # exactly once, which is what makes the gap list complete out to the radius.
    live_by_partition: defaultdict = defaultdict(list)
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
            live_by_partition.clear()
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
        cutoff = start - search_radius

        key = key_of(current_call)
        survivors: list[int] = []
        for i in live_by_partition[key]:
            prev_end = ends[i]
            if prev_end < cutoff:
                continue  # unreachable by this call and by every later one
            survivors.append(i)
            overlap_end = min(prev_end, end)

            if start < overlap_end:
                prev_size = prev_end - starts[i]
                ov_a.append(i)
                ov_b.append(current_call_index)
                ov_key.append((overlap_end - start) / max(prev_size, current_size))
            else:
                # start >= prev_end here, so the distance is non-negative, and
                # prev_end >= cutoff bounds it by search_radius.
                gap_a.append(i)
                gap_b.append(current_call_index)
                gap_key.append(start - prev_end)

        survivors.append(current_call_index)
        live_by_partition[key] = survivors

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
        partition_by=partition_by,
        search_radius=search_radius,
    )


CallSource = CallSet | Iterable[Call] | str | Path


def bed_paths_of(source: str | Path) -> list[Path]:
    """The BED files one path argument names: itself, or a glob pattern's matches.

    A path naming an existing file is taken as is, so a filename that happens to
    contain a glob character is never expanded.
    """
    if isinstance(source, str) and glob.has_magic(source) and not Path(source).is_file():
        matches = sorted(glob.glob(source))
        if not matches:
            raise FileNotFoundError(f"no files match {source!r}")
        return [Path(match) for match in matches]
    return [Path(source)]


def collect_callsets(
    sources: CallSource | Iterable[CallSource],
    *,
    chromosome_order: Iterable[str] | None = None,
    partition_by: Iterable[str] = PARTITION_FIELDS,
    search_radius: int = 0,
) -> CallSet:
    """Pool calls from any mix of BED paths, CallSets and Call iterables into one CallSet."""
    if isinstance(sources, (CallSet, str, Path)):
        sources = [sources]

    calls: list[Call] = []
    for source in sources:
        if isinstance(source, Call):
            calls.append(source)
        elif isinstance(source, CallSet):
            calls.extend(source.calls)
        elif isinstance(source, (str, Path)):
            for path in bed_paths_of(source):
                calls.extend(read_bed_calls(path))
        else:
            calls.extend(source)

    return build_callset(
        calls,
        chromosome_order=chromosome_order,
        partition_by=partition_by,
        search_radius=search_radius,
    )
