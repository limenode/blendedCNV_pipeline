"""Writing classified intervals back out as BED: the TP / FP / FN rows."""

from pathlib import Path

import numpy as np

from consensuscnv.callsets.bed_io import source_strings_for
from consensuscnv.callsets.registry import CHROMOSOMES, SAMPLES, SVTYPES
from consensuscnv.classification.classify import Classification, ClassLabel
from consensuscnv.classification.intervals import IntervalSet


def write_intervals_bed(
    intervals: IntervalSet,
    path: str | Path,
    *,
    include_sample: bool = True,
) -> int:
    """Write an IntervalSet to BED, returning the number of rows written.

    Columns are chrom, start, end, svtype, source, and optionally sample_id.
    Layout is consistent with `write_merged_bed`, so a label file and a
    consensus file can be diffed against each other.
    """
    chrom_names = CHROMOSOMES.names
    sample_names = SAMPLES.names
    svtype_names = SVTYPES.names

    # Sorted on registry ids for chrom, and on name for sample and svtype.
    sample_rank = np.argsort(np.argsort(sample_names))[intervals.sample_idx]
    svtype_rank = np.argsort(np.argsort(svtype_names))[intervals.svtype_idx]
    order = np.lexsort(
        (sample_rank, svtype_rank, intervals.ends, intervals.starts, intervals.chrom_idx)
    )

    chrom_ids = intervals.chrom_idx[order].tolist()
    svtype_ids = intervals.svtype_idx[order].tolist()
    starts = intervals.starts[order].tolist()
    ends = intervals.ends[order].tolist()
    sources = source_strings_for(intervals.source_bits[order].tolist())

    if include_sample:
        sample_ids = intervals.sample_idx[order].tolist()
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

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as bed_file:
        bed_file.writelines(rows)
    return len(rows)


def write_labels(
    classification: Classification,
    directory: str | Path,
    stem: str,
) -> dict[ClassLabel, Path]:
    """Write `<stem>.tp.bed`, `.fp.bed` and `.fn.bed`, returning their paths.

    TP and FP rows are query calls; FN rows are truth intervals nothing matched.
    A file is written even when empty.
    """
    directory = Path(directory)
    sides = {
        ClassLabel.TRUE_POSITIVE: classification.query,
        ClassLabel.FALSE_POSITIVE: classification.query,
        ClassLabel.FALSE_NEGATIVE: classification.truth,
    }

    written: dict[ClassLabel, Path] = {}
    for label, side in sides.items():
        path = directory / f"{stem}.{label.value.lower()}.bed"
        write_intervals_bed(side.select(classification.rows_for(label)), path)
        written[label] = path
    return written
