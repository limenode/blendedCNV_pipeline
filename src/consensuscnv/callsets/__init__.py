"""Interval call sets and their overlap structure. The pipeline's core primitive.
Interned ids (chrom_idx, svtype_idx, sample_idx, source_bits) come from
process-wide registries in `registry`, not from the CallSet.

    calls = read_bed_calls(path)                     # bed_io
    callset = collect_callsets([...])                # callset
    selection = filter_edges(callset, 0.5)           # edges
    merged = merge_components(callset, selection)    # merging
    write_merged_bed(merged, out)           # bed_io
"""

from consensuscnv.callsets.bed_io import (
    read_bed_calls,
    source_strings_for,
    write_merged_bed,
)
from consensuscnv.callsets.calls import Call
from consensuscnv.callsets.callset import (
    CallSet,
    CallSource,
    build_callset,
    collect_callsets,
    sort_into_genome_order,
)
from consensuscnv.callsets.edges import EdgeSelection, filter_edges
from consensuscnv.callsets.merging import MergedCallSet, merge_components
from consensuscnv.callsets.registry import DEFAULT_CHROMOSOME_ORDER, Registry

__all__ = [
    "DEFAULT_CHROMOSOME_ORDER",
    "Call",
    "CallSet",
    "CallSource",
    "EdgeSelection",
    "MergedCallSet",
    "Registry",
    "build_callset",
    "collect_callsets",
    "filter_edges",
    "merge_components",
    "read_bed_calls",
    "sort_into_genome_order",
    "source_strings_for",
    "write_merged_bed",
]
