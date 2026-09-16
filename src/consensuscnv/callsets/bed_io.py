"""Reading calls from BED and writing merged call sets back out."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from consensuscnv.callsets.calls import Call
from consensuscnv.callsets.merging import MergedCallSet
from consensuscnv.callsets.registry import CHROMOSOMES, SAMPLES, SOURCES, SVTYPES


def read_bed_calls(path: str | Path, *, sample_id: str | None = None) -> Iterator[Call]:
    """Yield Calls from a BED of ``chrom start end svtype source [sample_id]``.

    Both layouts this package writes read back: the per-sample five-column file,
    whose sample is the filename stem, and the combined six-column file from
    `write_merged_bed(include_sample=True)`, whose sample is the sixth column.
    An explicit `sample_id` overrides both.

    A consensus file's `source` column is the pipe-joined callers behind each
    call (``cnvpytor|delly``); it comes back as one opaque source label, so
    `n_sources` on a graph built from it counts labels rather than callers.
    """
    path = Path(path) if isinstance(path, str) else path
    stem = path.stem

    with open(path, "r") as bed_file:
        for line in bed_file:
            if line.startswith("#"):
                continue  # Skip comment lines
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue  # Skip lines that don't have enough fields

            yield Call(
                chrom=fields[0],
                start=int(fields[1]),
                end=int(fields[2]),
                svtype=fields[3],
                source=fields[4],
                sample_id=sample_id or (fields[5] if len(fields) > 5 else stem),
            )


def source_strings_for(masks: list[int]) -> list[str]:
    """Render each source bitmask as a pipe-joined name string."""
    names = SOURCES.names
    cache: dict[int, str] = {}
    rendered: list[str] = []
    for mask in masks:
        text = cache.get(mask)
        if text is None:
            text = cache[mask] = "|".join(
                sorted(name for i, name in enumerate(names) if mask >> i & 1)
            )
        rendered.append(text)
    return rendered


def write_merged_bed(
    merged: MergedCallSet,
    path: str | Path,
    *,
    include_sample: bool = False,
) -> int:
    """
    Write a merged call set to BED, returning the number of rows written.
    Columns are chrom, start, end, svtype, source, and optionally sample_id.

    The svtype column, and the sample column when asked for, are per-component
    values, so the parent CallSet must partition on those fields.
    """
    chrom_names = CHROMOSOMES.names
    sample_names = SAMPLES.names
    svtype_names = SVTYPES.names

    chrom_ids = merged.chrom_idx
    svtype_ids = merged.svtype_idx
    svtype_rank = np.argsort(np.argsort(svtype_names))[svtype_ids]
    if "sample_id" in merged.parent.partition_by:
        sample_ids = merged.sample_idx
        sample_rank = np.argsort(np.argsort(sample_names))[sample_ids]
    else:
        if include_sample:
            merged.parent.require_partition("sample_id")
        sample_ids = sample_rank = np.zeros(len(merged), dtype=np.int64)
    order = np.lexsort((sample_rank, svtype_rank, merged.ends, merged.starts, chrom_ids))

    chrom_ids = chrom_ids[order].tolist()
    svtype_ids = svtype_ids[order].tolist()
    starts = merged.starts[order].tolist()
    ends = merged.ends[order].tolist()
    sources = source_strings_for(merged.source_bits[order].tolist())


    if include_sample:
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
