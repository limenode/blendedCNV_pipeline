"""Consensus calling: per-tool BEDs in, one consensus BED per level and overlap out.

It parses the experimental VCFs, merges each call set, and writes every consensus
level.

  * `build_callset` records every joinable pair sorted by reciprocal overlap, so
    `filter_edges` is a `searchsorted` plus a slice -- a view, not a rebuild. One
    built CallSet serves every overlap.
  * `min_sources` is a post-hoc component filter, so one `merge_components` at the
    default `min_sources=1` followed by `n_sources >= k` reproduces every level
    exactly.
"""

from __future__ import annotations

import glob
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from consensuscnv.callsets import (
    collect_callsets,
    filter_edges,
    merge_components,
    read_bed_calls,
    write_merged_bed,
)
from consensuscnv.callsets.merging import MergedCallSet
from consensuscnv.callsets.registry import SAMPLES, seed_chromosomes
from consensuscnv.utils import PipelineConfig, load_sample_list


@dataclass(frozen=True)
class ConsensusRun:
    """One written consensus set: a (call set, overlap, level) point.

    `paths` holds one BED for the combined default and one per sample under
    `--per-sample`.
    """

    call_set: str
    reciprocal_overlap: float
    level: int
    n_sources: int
    n_calls: int
    paths: tuple[Path, ...]

    @property
    def label(self) -> str:
        return f"{self.level}of{self.n_sources}"


@dataclass(frozen=True)
class ConsensusSet:
    """A consensus set.

    `run_consensus` writes these; the benchmark path classifies them. Sharing the
    generator keeps the two from drifting, and means the benchmark path never has
    to read its own output back off disk.
    """

    call_set: str
    reciprocal_overlap: float
    level: int
    n_sources: int
    merged: MergedCallSet

    @property
    def label(self) -> str:
        return f"{self.level}of{self.n_sources}"


def tool_bed_paths(
    config: PipelineConfig, call_set: str, samples: frozenset[str] | None = None
) -> list[str]:
    """Every per-sample BED written for `call_set`, one tool directory at a time.

    `samples` drops BEDs outside the allowlist.
    """
    layout = config.layout
    paths = [
        bed
        for tool in config.experimental[call_set]
        for bed in sorted(glob.glob(str(layout.bed_tool_dir(call_set, tool) / "*.bed")))
    ]
    if samples is None:
        return paths
    return [path for path in paths if Path(path).stem in samples]


def parse_experimental(config: PipelineConfig) -> None:
    """Parse the experimental VCFs into per-tool BED files.

    Only the experimental sets: controls and benchmarks belong to the evaluation
    path.
    """
    from consensuscnv.parsing.parser_utils import ExclusionMask
    from consensuscnv.parsing.vcf_parser import process_vcfs_to_beds

    print("\nParsing experimental datasets...")
    process_vcfs_to_beds(
        config,
        ExclusionMask.load(config.excluded_regions_file),
        samples=load_sample_list(config.sample_list_file),
    )


def iter_consensus_sets(config: PipelineConfig) -> Iterator[ConsensusSet]:
    """Every (call set, overlap, level) consensus set, merged in memory.

    Assumes the per-tool BEDs are already on disk; call `parse_experimental` first
    if they are not.
    """
    # Before any build_callset, so chrom ids order the genome rather than the data.
    seed_chromosomes(config.chromosomes)
    params = config.consensus
    samples = load_sample_list(config.sample_list_file)

    for call_set, tools in config.experimental.items():
        paths = tool_bed_paths(config, call_set, samples)
        if not paths:
            print(f"\nNo BED files found for {call_set!r}; skipping.")
            continue

        print(f"\nMerging {call_set!r}: {len(paths)} BED files from {len(tools)} tools")
        calls = collect_callsets(
            (read_bed_calls(path) for path in paths), chromosome_order=config.chromosomes
        )
        n_sources = len(tools)

        for overlap in params.reciprocal_overlaps:
            # A view over the prebuilt edge lists, not another graph build.
            merged = merge_components(
                calls, filter_edges(calls, min_reciprocal_overlap=overlap)
            )
            above_floor = (merged.ends - merged.starts) >= params.min_size
            levels = merged.n_sources  # computed once, not once per level

            for level in range(1, n_sources + 1):
                yield ConsensusSet(
                    call_set=call_set,
                    reciprocal_overlap=overlap,
                    level=level,
                    n_sources=n_sources,
                    merged=merged.select(above_floor & (levels >= level)),
                )


def run_consensus(
    config: PipelineConfig,
    *,
    reuse_beds: bool = False,
    per_sample: bool = False,
) -> list[ConsensusRun]:
    """Parse, merge and write every consensus level for every call set.

    With `reuse_beds`, the per-tool BEDs already under `output_dir` are used as-is
    and no VCF is read. `per_sample` splits each output into one BED per sample
    instead of one carrying a sample column.
    """
    if not reuse_beds:
        parse_experimental(config)

    return [
        _write_set(config, consensus_set, per_sample)
        for consensus_set in iter_consensus_sets(config)
    ]


def _write_set(
    config: PipelineConfig, consensus_set: ConsensusSet, per_sample: bool
) -> ConsensusRun:
    """Write one (call set, overlap, level) point and report what it held."""
    layout = config.layout
    selected = consensus_set.merged
    call_set = consensus_set.call_set
    overlap = consensus_set.reciprocal_overlap
    level, n_sources = consensus_set.level, consensus_set.n_sources

    def run(n_calls: int, paths: tuple[Path, ...]) -> ConsensusRun:
        return ConsensusRun(call_set, overlap, level, n_sources, n_calls, paths)

    if not per_sample:
        path = layout.consensus_bed(call_set, overlap, level, n_sources)
        path.parent.mkdir(parents=True, exist_ok=True)
        return run(write_merged_bed(selected, path, include_sample=True), (path,))

    sample_ids = selected.sample_idx
    names = SAMPLES.names
    written: list[Path] = []
    n_calls = 0
    for sample_idx in np.unique(sample_ids):
        path = layout.consensus_sample_bed(
            call_set, overlap, level, n_sources, names[sample_idx]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        n_calls += write_merged_bed(selected.select(sample_ids == sample_idx), path)
        written.append(path)
    return run(n_calls, tuple(written))


def format_summary(runs: list[ConsensusRun]) -> str:
    """A call set x level table of consensus call counts, for the run summary."""
    if not runs:
        return "No consensus sets were written."

    lines = ["", f"{'Call set':<20}{'Overlap':>9}{'Level':>8}{'Calls':>10}"]
    for run in runs:
        lines.append(
            f"{run.call_set:<20}{run.reciprocal_overlap:>9.2f}"
            f"{run.label:>8}{run.n_calls:>10,}"
        )
    return "\n".join(lines)
