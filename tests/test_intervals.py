"""The one-call constructors: files or plain records in, an IntervalSet with its
graph out. Everything they accept is routed through `collect_callsets`, so this
is also where its path handling is pinned down.
"""

import numpy as np
import pytest

from consensuscnv.callsets import collect_callsets
from consensuscnv.callsets.registry import SAMPLES
from consensuscnv.classification import IntervalSet, build_candidates

ROWS = {
    "S1": ["chr1\t1000\t3000\tDEL\tcnvpytor\n", "chr2\t5000\t7000\tDUP\tdelly\n"],
    "S2": ["chr1\t1100\t3100\tDEL\tgatk\n"],
}


@pytest.fixture
def per_sample_beds(tmp_path):
    """Two five-column BEDs in one directory, the pipeline's per-sample layout."""
    directory = tmp_path / "caller"
    directory.mkdir()
    for sample, rows in ROWS.items():
        (directory / f"{sample}.bed").write_text("".join(rows))
    return directory


def sample_names(intervals):
    return sorted({SAMPLES.names[i] for i in intervals.sample_idx})


def test_from_bed_takes_one_path_a_list_or_a_glob(per_sample_beds):
    one = IntervalSet.from_bed(per_sample_beds / "S1.bed")
    assert len(one) == 2 and sample_names(one) == ["S1"]

    listed = IntervalSet.from_bed([per_sample_beds / "S1.bed", str(per_sample_beds / "S2.bed")])
    globbed = IntervalSet.from_bed(str(per_sample_beds / "*.bed"))
    assert len(listed) == len(globbed) == 3
    assert sample_names(listed) == sample_names(globbed) == ["S1", "S2"]
    assert np.array_equal(listed.starts, globbed.starts)

    with pytest.raises(FileNotFoundError, match="no files match"):
        IntervalSet.from_bed(str(per_sample_beds / "*.vcf"))


def test_a_bare_string_is_one_path_not_a_sequence_of_characters(per_sample_beds):
    """`str` is iterable; treating it as one would read files named "S", "1", ..."""
    callset = collect_callsets(str(per_sample_beds / "S1.bed"))
    assert len(callset) == 2


def test_from_bed_carries_the_graph_and_the_build_arguments(per_sample_beds, seeded_registry):
    """The origin CallSet is the built graph, under whatever partition was asked for."""
    within = IntervalSet.from_bed(str(per_sample_beds / "*.bed"), genome=seeded_registry)
    assert within.origin.partition_by == ("svtype", "sample_id")
    assert len(within.origin.ov_a) == 0            # S1 and S2 overlap, but never within a sample

    across = IntervalSet.from_bed(
        str(per_sample_beds / "*.bed"), partition_by=("svtype",), search_radius=500
    )
    assert across.origin.partition_by == ("svtype",)
    assert across.origin.search_radius == 500
    assert len(across.origin.ov_a) == 1            # the chr1 deletion, shared by both samples

    # And the two sides of a comparison come straight off it.
    s1 = across.select(np.array([n == "S1" for n in (SAMPLES.names[i] for i in across.sample_idx)]))
    s2 = across.select(np.array([n == "S2" for n in (SAMPLES.names[i] for i in across.sample_idx)]))
    assert len(build_candidates(s1, s2, partition_by=("svtype",)).ov_q) == 1


def test_from_records_fills_defaults_and_builds_the_same_graph(per_sample_beds):
    from_files = IntervalSet.from_bed(str(per_sample_beds / "*.bed"), partition_by=("svtype",))
    from_rows = IntervalSet.from_records(
        [
            ("chr1", 1000, 3000, "DEL", "cnvpytor", "S1"),
            ("chr2", 5000, 7000, "DUP", "delly", "S1"),
            {"chrom": "chr1", "start": 1100, "end": 3100, "sample_id": "S2"},
        ],
        svtype="DEL",
        source="gatk",
        partition_by=("svtype",),
    )
    assert np.array_equal(from_files.starts, from_rows.starts)
    assert np.array_equal(from_files.ends, from_rows.ends)
    assert np.array_equal(from_files.svtype_idx, from_rows.svtype_idx)
    assert np.array_equal(from_files.sample_idx, from_rows.sample_idx)
    assert np.array_equal(from_files.origin.ov_key, from_rows.origin.ov_key)


def test_names_are_the_registry_lookups_of_the_id_columns(per_sample_beds):
    intervals = IntervalSet.from_bed(str(per_sample_beds / "*.bed"))
    assert intervals.chrom_names.tolist() == ["chr1", "chr1", "chr2"]
    assert intervals.svtype_names.tolist() == ["DEL", "DEL", "DUP"]
    assert intervals.sample_names.tolist() == ["S1", "S2", "S1"]
    assert intervals.source_names == ["cnvpytor", "gatk", "delly"]
    assert intervals.samples == ["S1", "S2"]


def test_restrict_to_samples_takes_names_or_ids(per_sample_beds):
    intervals = IntervalSet.from_bed(str(per_sample_beds / "*.bed"))
    by_name = intervals.restrict_to_samples(["S1"])
    by_id = intervals.restrict_to_samples([SAMPLES.get("S1")])
    assert by_name.samples == by_id.samples == ["S1"] and len(by_name) == len(by_id) == 2

    rest = intervals.restrict_to_samples(["S1"], invert=True)
    assert rest.samples == ["S2"] and len(rest) == 1
    assert len(intervals.restrict_to_samples(["never-seen"])) == 0
    assert len(intervals.restrict_to_samples(["never-seen"], invert=True)) == len(intervals)


def test_to_frame_is_the_bed_layout_by_name(per_sample_beds):
    frame = IntervalSet.from_bed(str(per_sample_beds / "*.bed")).to_frame()
    assert list(frame.columns) == ["chrom", "start", "end", "svtype", "source", "sample_id", "n_sources"]
    assert frame.iloc[1].tolist() == ["chr1", 1100, 3100, "DEL", "gatk", "S2", 1]


def test_classification_frames_carry_labels(per_sample_beds):
    from consensuscnv.classification import classify

    intervals = IntervalSet.from_bed(str(per_sample_beds / "*.bed"))
    s1, s2 = intervals.restrict_to_samples(["S1"]), intervals.restrict_to_samples(["S2"])
    with pytest.warns(UserWarning, match=r"1 truth samples vs. 1 query samples.*\['S2'\]"):
        result = classify(build_candidates(s1, s2, partition_by=("svtype",)), min_reciprocal_overlap=0.5)

    query = result.to_frame()
    assert query["label"].tolist() == ["TP", "FP"]            # the chr1 DEL matches S2's; the DUP has no partner
    assert query["n_partners"].tolist() == [1, 0]
    truth = result.to_frame("truth")
    assert truth["found"].tolist() == [True] and truth["n_partners"].tolist() == [1]
    with pytest.raises(ValueError, match="side must be"):
        result.to_frame("both")
