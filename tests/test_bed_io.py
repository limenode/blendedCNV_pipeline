"""BED I/O: row order comes from the genome, a write survives a read, and plain
records become Calls."""

import pytest

from consensuscnv.callsets import (
    Call,
    calls_from_records,
    merge_components,
    read_bed_calls,
    write_merged_bed,
)
from consensuscnv.callsets.registry import CHROMOSOMES, SAMPLES


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


def test_bed_round_trip_preserves_intervals_and_samples(callset, tmp_path):
    """Writing and reading back must not lose or alter an interval -- or its sample.

    The combined file carries the sample in column 6, and a reader taking it
    from the filename instead would assign every call to a sample called
    "merged", which is exactly what happened before the column was read.
    """
    merged = merge_components(callset, min_reciprocal_overlap=0.5)
    path = tmp_path / "merged.bed"
    write_merged_bed(merged, path, include_sample=True)

    round_trip = sorted(
        (call.chrom, call.start, call.end, call.sample_id) for call in read_bed_calls(path)
    )
    expected = sorted(
        (CHROMOSOMES.names[merged.parent.chrom_idx[row]], int(start), int(end), SAMPLES.names[s])
        for row, start, end, s in zip(
            merged.representative, merged.starts, merged.ends, merged.sample_idx
        )
    )
    assert round_trip == expected
    assert {call.sample_id for call in read_bed_calls(path)} == {"S1", "S2"}


def test_sample_comes_from_the_filename_or_the_caller_when_no_column(tmp_path):
    """Five columns: the stem names the sample. An explicit `sample_id` wins over both."""
    five = tmp_path / "NA12878.bed"
    five.write_text("chr1\t100\t200\tDEL\tcnvpytor\n")
    assert [call.sample_id for call in read_bed_calls(five)] == ["NA12878"]
    assert [call.sample_id for call in read_bed_calls(five, sample_id="X")] == ["X"]

    six = tmp_path / "2of3.bed"
    six.write_text("chr1\t100\t200\tDEL\tcnvpytor|delly\tNA12878\n")
    (call,) = read_bed_calls(six)
    assert call.sample_id == "NA12878" and call.source == "cnvpytor|delly"
    assert [call.sample_id for call in read_bed_calls(six, sample_id="X")] == ["X"]


def test_calls_from_records_fills_in_what_a_record_leaves_out():
    """Tuples of 3 to 6 fields and mappings both work, and defaults apply per field."""
    calls = list(
        calls_from_records(
            [
                ("chr1", 100, 200),
                ("chr1", 300, 400, "DUP"),
                ("chr1", 500, 600, "DEL", "delly", "S9"),
                {"chrom": "chr2", "start": "10", "end": "20", "sample_id": "S1"},
            ],
            source="array",
        )
    )
    assert calls[0] == Call("chr1", 100, 200, "CNV", "array", "sample")
    assert calls[1] == Call("chr1", 300, 400, "DUP", "array", "sample")
    assert calls[2] == Call("chr1", 500, 600, "DEL", "delly", "S9")
    assert calls[3] == Call("chr2", 10, 20, "CNV", "array", "S1")
    with pytest.raises(ValueError, match="3 to 6 fields"):
        list(calls_from_records([("chr1", 100)]))
