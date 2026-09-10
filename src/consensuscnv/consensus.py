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
from consensuscnv.callsets.registry import SAMPLES, seed_chromosomes
from consensuscnv.parsing.parser_utils import ExclusionMask, load_sample_list
from consensuscnv.parsing.vcf_parser import process_vcfs_to_beds
from consensuscnv.utils import PipelineConfig


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


def tool_bed_paths(config: PipelineConfig, call_set: str) -> list[str]:
    """Every per-sample BED written for `call_set`, one tool directory at a time.

    Pinned to the tool labels the config names rather than globbing the call set
    directory, so nothing that later lands beside them is read back in as an extra
    source. `layout.consensus` is outside this tree for the same reason.
    """
    layout = config.layout
    return [
        bed
        for tool in config.experimental[call_set]
        for bed in sorted(glob.glob(str(layout.bed_tool_dir(call_set, tool) / "*.bed")))
    ]


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
    # Before any build_callset, so chrom ids order the genome rather than the data.
    seed_chromosomes(config.chromosomes)

    if not reuse_beds:
        print("\nParsing experimental datasets...")
        process_vcfs_to_beds(
            config,
            ExclusionMask.load(config.excluded_regions_file),
            samples=load_sample_list(config.sample_list_file),
        )

    params = config.consensus
    runs: list[ConsensusRun] = []

    for call_set, tools in config.experimental.items():
        paths = tool_bed_paths(config, call_set)
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
                keep = above_floor & (levels >= level)
                runs.append(
                    _write_level(
                        config, merged, keep, call_set, overlap, level, n_sources, per_sample
                    )
                )

    return runs


def _write_level(
    config: PipelineConfig,
    merged,
    keep: np.ndarray,
    call_set: str,
    overlap: float,
    level: int,
    n_sources: int,
    per_sample: bool,
) -> ConsensusRun:
    """Write one (call set, overlap, level) point and report what it held."""
    layout = config.layout
    selected = merged.select(keep)

    if not per_sample:
        path = layout.consensus_bed(call_set, overlap, level, n_sources)
        path.parent.mkdir(parents=True, exist_ok=True)
        n_calls = write_merged_bed(selected, path, include_sample=True)
        return ConsensusRun(call_set, overlap, level, n_sources, n_calls, (path,))

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
    return ConsensusRun(call_set, overlap, level, n_sources, n_calls, tuple(written))


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
