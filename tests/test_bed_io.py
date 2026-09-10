"""BED output: row order comes from the genome, and a write survives a read."""

from consensuscnv.callsets import merge_components, read_bed_calls, write_merged_bed
from consensuscnv.callsets.registry import CHROMOSOMES


def test_written_rows_follow_genome_order(callset, seeded_registry, tmp_path):
    """chr2 was handed over first; the file must still start with chr1.

    `write_merged_bed` sorts on `chrom_idx` -- the registry id, not the name -- so
    this is really a test that `seed_chromosomes` made the id space a property of
    the genome file rather than of the order the data arrived in.
    """
    merged = merge_components(callset, min_reciprocal_overlap=0.5)
    path = tmp_path / "merged.bed"

    n_written = write_merged_bed(merged, path, include_sample=True)
    assert n_written == len(merged)

    seen = list(dict.fromkeys(line.split("\t")[0] for line in path.read_text().splitlines()))
    assert seen == list(seeded_registry)


def test_bed_round_trip_preserves_intervals(callset, tmp_path):
    """Writing and reading back must not lose or alter an interval."""
    merged = merge_components(callset, min_reciprocal_overlap=0.5)
    path = tmp_path / "merged.bed"
    write_merged_bed(merged, path, include_sample=True)

    round_trip = sorted((call.chrom, call.start, call.end) for call in read_bed_calls(path))
    expected = sorted(
        (CHROMOSOMES.names[merged.parent.chrom_idx[row]], int(start), int(end))
        for row, start, end in zip(merged.representative, merged.starts, merged.ends)
    )
    assert round_trip == expected
