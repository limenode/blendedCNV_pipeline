"""What a CallSet's `partition_by` and `search_radius` guarantee.

Both are build-time choices recorded on the object, and the rest of the package
either relies on them (every per-component chrom / svtype / sample read) or is
bounded by them (every `max_padding` filter). Nothing else checks that the
guarantees hold or that the guards fire.
"""

import numpy as np
import pytest

from consensuscnv.callsets import (
    Call,
    build_callset,
    collect_callsets,
    filter_edges,
    merge_components,
    write_merged_bed,
)
from consensuscnv.classification.intervals import IntervalSet
from consensuscnv.classification.pairs import build_candidates, filter_candidates


def sample_of(callset, rows):
    return callset.sample_idx[rows]


# --------------------------------------------------------------------------- #
# partition_by
# --------------------------------------------------------------------------- #
def test_default_partition_never_joins_samples(callset):
    """The consensus graph must not contain a single cross-sample edge."""
    assert callset.partition_by == ("svtype", "sample_id")
    assert np.array_equal(sample_of(callset, callset.ov_a), sample_of(callset, callset.ov_b))
    assert np.array_equal(sample_of(callset, callset.gap_a), sample_of(callset, callset.gap_b))


def test_dropping_sample_from_the_partition_joins_samples(seeded_registry, synthetic_calls):
    """S1 and S2 share the chr1 1-3 kb deletion; only a cross-sample graph sees it."""
    cross = collect_callsets(
        [synthetic_calls], chromosome_order=seeded_registry, partition_by=("svtype",)
    )
    assert cross.partition_by == ("svtype",)
    crossing = sample_of(cross, cross.ov_a) != sample_of(cross, cross.ov_b)
    assert crossing.any()
    # svtype still partitions: no edge joins a DEL to a DUP.
    assert np.array_equal(cross.svtype_idx[cross.ov_a], cross.svtype_idx[cross.ov_b])

    merged = merge_components(cross, min_reciprocal_overlap=0.5)
    # The six chr1 1-3 kb calls (two samples x three callers) become one locus.
    assert merged.n_calls.max() == 6


def test_partition_is_canonicalised_and_validated(seeded_registry, synthetic_calls):
    reordered = build_callset(
        synthetic_calls, chromosome_order=seeded_registry, partition_by=["sample_id", "svtype"]
    )
    assert reordered.partition_by == ("svtype", "sample_id")
    with pytest.raises(ValueError, match="partition_by names"):
        build_callset(synthetic_calls, chromosome_order=seeded_registry, partition_by=("chrom",))


def test_per_component_reads_refuse_a_field_that_no_longer_partitions(
    seeded_registry, synthetic_calls, tmp_path
):
    """A component spanning samples has no sample; every reader must say so."""
    cross = collect_callsets(
        [synthetic_calls], chromosome_order=seeded_registry, partition_by=("svtype",)
    )
    merged = merge_components(cross, min_reciprocal_overlap=0.5)

    assert len(merged.chrom_idx) == len(merged)      # chrom always partitions
    assert len(merged.svtype_idx) == len(merged)     # still in the partition
    with pytest.raises(ValueError, match="sample_id"):
        _ = merged.sample_idx
    with pytest.raises(ValueError, match="sample_id"):
        IntervalSet.from_merged(merged)
    with pytest.raises(ValueError, match="sample_id"):
        write_merged_bed(merged, tmp_path / "with_sample.bed", include_sample=True)

    # Without a sample column the set is still writable.
    assert write_merged_bed(merged, tmp_path / "loci.bed") == len(merged)


def test_candidates_mirror_the_callset_partition(seeded_registry, synthetic_calls):
    """The bipartite route and the one-graph route must agree on isolation.

    A call in the query set is "isolated" at a threshold when no truth-set call
    clears it. Counting that off `build_candidates(partition_by=("svtype",))`
    and off a cross-sample CallSet has to give the same number at every
    threshold, since both use the same reciprocal-overlap key.
    """
    within = collect_callsets([synthetic_calls], chromosome_order=seeded_registry)
    intervals = IntervalSet.from_callset(within)
    is_s1 = np.array([c.sample_id == "S1" for c in within.calls])
    query, other = intervals.select(is_s1), intervals.select(~is_s1)

    default = build_candidates(query, other)
    assert len(default) == 0                       # sample in the partition: nothing crosses
    crossing = build_candidates(query, other, partition_by=("svtype",))
    assert crossing.partition_by == ("svtype",)
    assert len(crossing.ov_q) > 0

    cross = collect_callsets(
        [synthetic_calls], chromosome_order=seeded_registry, partition_by=("svtype",)
    )
    s1 = np.array([c.sample_id == "S1" for c in cross.calls])
    for threshold in (0.05, 0.5, 0.99):
        pairs = filter_candidates(crossing, min_reciprocal_overlap=threshold)
        isolated_bipartite = len(query) - len(np.unique(pairs.query_row))

        edges = filter_edges(cross, min_reciprocal_overlap=threshold)
        a, b = edges.a, edges.b
        covered = np.zeros(len(cross), dtype=bool)
        covered[a[s1[a] & ~s1[b]]] = True
        covered[b[s1[b] & ~s1[a]]] = True
        isolated_graph = int((s1 & ~covered).sum())

        assert isolated_bipartite == isolated_graph


# --------------------------------------------------------------------------- #
# search_radius
# --------------------------------------------------------------------------- #
def gapped(*intervals, search_radius):
    calls = [Call("chr1", start, end, "DEL", "x", "S") for start, end in intervals]
    return build_callset(calls, chromosome_order=("chr1",), search_radius=search_radius)


def test_gap_edges_are_complete_out_to_the_radius():
    """Every pair within the radius is recorded, whether or not a chain links them.

    A=[0,10], B=[20,30], C=[40,50]: A-B and B-C are 10 bp apart, A-C 30 bp. The
    A-C pair is only reachable by comparing across the gap at B.
    """
    callset = gapped((0, 10), (20, 30), (40, 50), search_radius=30)
    assert sorted(callset.gap_key.tolist()) == [10, 10, 30]
    assert len(filter_edges(callset, max_padding=30)) == 3
    assert len(filter_edges(callset, max_padding=10)) == 2

    # Below the radius the pair is not recorded, and the filter refuses to guess.
    narrow = gapped((0, 10), (20, 30), (40, 50), search_radius=10)
    assert sorted(narrow.gap_key.tolist()) == [10, 10]
    with pytest.raises(ValueError, match="search_radius"):
        filter_edges(narrow, max_padding=30)


def test_radius_zero_records_touching_pairs_only():
    """`max_padding=0` bridges exactly-touching intervals, so distance 0 is kept."""
    callset = gapped((0, 10), (10, 20), (25, 30), search_radius=0)
    assert callset.gap_key.tolist() == [0]
    assert len(merge_components(callset, max_padding=0)) == 2
    assert len(merge_components(callset, max_padding=None)) == 3


def test_components_do_not_depend_on_radius_beyond_the_padding(seeded_registry, synthetic_calls):
    """Widening the radius adds edges above the padding and must change nothing below it."""
    narrow = build_callset(synthetic_calls, chromosome_order=seeded_registry, search_radius=1_000)
    wide = build_callset(synthetic_calls, chromosome_order=seeded_registry, search_radius=100_000)
    assert len(wide.gap_a) > len(narrow.gap_a)
    for padding in (0, 500, 1_000):
        a = merge_components(narrow, max_padding=padding)
        b = merge_components(wide, max_padding=padding)
        assert np.array_equal(a.labels, b.labels)
        assert np.array_equal(a.starts, b.starts) and np.array_equal(a.ends, b.ends)


def test_overlap_edges_ignore_the_radius(seeded_registry, synthetic_calls):
    """Overlapping pairs are recorded at every radius; only gap pairs are bounded."""
    zero = build_callset(synthetic_calls, chromosome_order=seeded_registry, search_radius=0)
    wide = build_callset(synthetic_calls, chromosome_order=seeded_registry, search_radius=10**6)
    assert np.array_equal(zero.ov_key, wide.ov_key)
    assert np.array_equal(zero.ov_a, wide.ov_a) and np.array_equal(zero.ov_b, wide.ov_b)
    with pytest.raises(ValueError, match="search_radius"):
        build_callset(synthetic_calls, chromosome_order=seeded_registry, search_radius=-1)
