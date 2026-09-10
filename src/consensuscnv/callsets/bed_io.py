"""Reading calls from BED and writing merged call sets back out."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from consensuscnv.callsets.calls import Call
from consensuscnv.callsets.merging import MergedCallSet
from consensuscnv.callsets.registry import CHROMOSOMES, SAMPLES, SOURCES, SVTYPES


def read_bed_calls(path: str | Path) -> Iterator[Call]:
    """Yield Calls from a 5-column BED. The sample id comes from the filename."""
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
    """
    chrom_names = CHROMOSOMES.names
    sample_names = SAMPLES.names
    svtype_names = SVTYPES.names

    representative = merged.representative
    parent = merged.parent
    chrom_ids = parent.chrom_idx[representative]
    svtype_ids = parent.svtype_idx[representative]
    sample_ids = parent.sample_idx[representative]

    sample_rank = np.argsort(np.argsort(sample_names))[sample_ids]
    svtype_rank = np.argsort(np.argsort(svtype_names))[svtype_ids]
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
