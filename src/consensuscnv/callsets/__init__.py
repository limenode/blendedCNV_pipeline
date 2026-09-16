"""Interval call sets and their overlap structure. The pipeline's core primitive.
Interned ids (chrom_idx, svtype_idx, sample_idx, source_bits) come from
process-wide registries in `registry`, not from the CallSet.

    seed_chromosomes(read_genome_file(genome_file))  # registry, once, first
    calls = read_bed_calls(path)                     # bed_io; or calls_from_records
    callset = collect_callsets([...])                # callset
    selection = filter_edges(callset, 0.5)           # edges
    merged = merge_components(callset, selection)    # merging
    write_merged_bed(merged, out)                    # bed_io

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
    build_callset,
    collect_callsets,
    sort_into_genome_order,
)
from consensuscnv.callsets.edges import EdgeSelection, filter_edges
from consensuscnv.callsets.merging import MergedCallSet, merge_components
from consensuscnv.callsets.registry import Registry, read_genome_file, seed_chromosomes

__all__ = [
    "PARTITION_FIELDS",
    "Call",
    "CallSet",
    "CallSource",
    "EdgeSelection",
    "MergedCallSet",
    "Registry",
    "build_callset",
    "calls_from_records",
    "collect_callsets",
    "filter_edges",
    "merge_components",
    "read_bed_calls",
    "read_genome_file",
    "seed_chromosomes",
    "sort_into_genome_order",
    "source_strings_for",
    "write_merged_bed",
]
