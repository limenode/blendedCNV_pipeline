"""Interval call sets and their overlap structure. The pipeline's core primitive.
Interned ids (chrom_idx, svtype_idx, sample_idx, source_bits) come from
process-wide registries in `registry`, not from the CallSet.

    seed_chromosomes(genome_file)                    # registry, once, first
    callset = collect_callsets("beds/*.bed")         # callset; paths, Calls or CallSets
    selection = filter_edges(callset, 0.5)           # edges
    merged = merge_components(callset, selection)    # merging
    write_merged_bed(merged, out)                    # bed_io

`IntervalSet.from_bed` in `consensuscnv.classification` wraps the first two
lines for anyone who wants the intervals and their graph in one call.

Edges never cross chromosome, nor the fields named by `partition_by` -- by
default svtype and sample, which is what consensus calling needs. Pass
`partition_by=("svtype",)` for a graph whose edges may join samples.
"""

from consensuscnv.callsets.bed_io import (
    read_bed_calls,
    source_strings_for,
    write_merged_bed,
)
from consensuscnv.callsets.calls import PARTITION_FIELDS, Call, calls_from_records
from consensuscnv.callsets.callset import (
    CallSet,
    CallSource,
    bed_paths_of,
    build_callset,
    collect_callsets,
    sort_into_genome_order,
)
from consensuscnv.callsets.edges import EdgeSelection, filter_edges
from consensuscnv.callsets.merging import MergedCallSet, merge_components
from consensuscnv.callsets.registry import (
    Registry,
    read_genome_file,
    reset_registries,
    seed_chromosomes,
)

__all__ = [
    "PARTITION_FIELDS",
    "Call",
    "CallSet",
    "CallSource",
    "EdgeSelection",
    "MergedCallSet",
    "Registry",
    "bed_paths_of",
    "build_callset",
    "calls_from_records",
    "collect_callsets",
    "filter_edges",
    "merge_components",
    "read_bed_calls",
    "read_genome_file",
    "reset_registries",
    "seed_chromosomes",
    "sort_into_genome_order",
    "source_strings_for",
    "write_merged_bed",
]
