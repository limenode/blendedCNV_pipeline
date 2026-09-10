"""Invariants of the merge layer that the consensus driver is built on.

`iter_consensus_sets` produces every consensus level from a *single*
`merge_components` call, and writes each level by row-selecting that one merge.
Both halves of that have to hold, and nothing else in the codebase checks them.
"""

import numpy as np
import pytest

from consensuscnv.callsets import filter_edges, merge_components

# Defined here rather than imported from conftest: `parametrize` needs the values
# at collection time, so they cannot come from a fixture, and importing across
# test modules is what conftest exists to avoid.
LEVELS = (1, 2, 3)
MERGE_COLUMNS = ("starts", "ends", "source_bits", "n_calls", "component_id")


@pytest.mark.parametrize("overlap", [0.0, 0.5])
@pytest.mark.parametrize("level", LEVELS)
def test_one_merge_shortcut_matches_an_explicit_merge(callset, overlap, level):
    """Selecting `n_sources >= k` off one merge must equal `min_sources=k`.

    This is what makes the consensus grid cheap: one merge per overlap serves
    every level. Checked at both overlaps because the edge selection differs
    between them and the shortcut has to hold for whichever was used.
    """
    once = merge_components(callset, filter_edges(callset, min_reciprocal_overlap=overlap))
    shortcut = once.select(once.n_sources >= level)
    explicit = merge_components(callset, min_reciprocal_overlap=overlap, min_sources=level)

    assert len(shortcut) == len(explicit)
    for column in MERGE_COLUMNS:
        assert np.array_equal(getattr(shortcut, column), getattr(explicit, column)), column


def test_select_keeps_labels_spanning_the_parent(callset):
    """`select` subsets rows; `labels` and `parent` must come through intact.

    `labels` maps every node of the *parent* CallSet to its component, so it is
    not indexed by this view's rows. Subsetting it alongside them is the obvious
    mistake and it is what this guards.
    """
    merged = merge_components(callset, min_reciprocal_overlap=0.5)
    keep = merged.n_sources >= 2
    subset = merged.select(keep)

    assert subset.parent is merged.parent, "select must not copy or replace the parent"
    # Contents and length rather than identity: a correct implementation is free
    # to copy, and failing it for that would be testing the implementation.
    assert len(subset.labels) == len(merged.parent.calls), (
        "labels index the parent's nodes, not the view's rows"
    )
    assert np.array_equal(subset.labels, merged.labels), "labels must be unchanged"
    assert np.array_equal(subset.starts, merged.starts[keep])
    assert np.array_equal(subset.component_id, merged.component_id[keep])


def test_consensus_levels_nest(callset):
    """3-of-N must be a subset of 2-of-N, and 2-of-N of 1-of-N.

    The paper leans on this: the highest-confidence calls are retained by
    stopping one level short, which is only true if the levels nest.
    """
    merged = merge_components(callset, min_reciprocal_overlap=0.5)
    ids = {
        level: set(merged.select(merged.n_sources >= level).component_id.tolist())
        for level in LEVELS
    }
    assert ids[3] <= ids[2] <= ids[1]
    assert len(ids[1]) > len(ids[2]) > len(ids[3]) > 0, "the fixture must exercise every level"


def test_min_sources_counts_callers_not_calls(callset):
    """A component with three calls from two callers is 2-of-N, not 3-of-N.

    `min_calls` counts member calls and `min_sources` counts distinct callers;
    consensus wants the latter. The fixture holds one component where CNVpytor
    reports the same event twice, which is the only place the two differ.
    """
    merged = merge_components(callset, min_reciprocal_overlap=0.5)
    duplicated = merged.select((merged.n_calls == 3) & (merged.n_sources == 2))
    assert len(duplicated) == 1, "the fixture must hold one two-caller, three-call component"

    component = int(duplicated.component_id[0])
    at_level = {
        level: set(merged.select(merged.n_sources >= level).component_id.tolist())
        for level in LEVELS
    }
    assert component in at_level[2], "two distinct callers means it reaches 2-of-N"
    assert component not in at_level[3], (
        "three calls from two callers must not reach 3-of-N -- that would be "
        "min_calls behaviour, not min_sources"
    )

    # And the same through the `min_sources=` parameter itself, not only through a
    # row selection off a default merge, so the merge's own filter is guarded.
    explicit = merge_components(callset, min_reciprocal_overlap=0.5, min_sources=3)
    assert component not in set(explicit.component_id.tolist())
    assert component in set(
        merge_components(callset, min_reciprocal_overlap=0.5, min_sources=2)
        .component_id.tolist()
    )


def test_edges_do_not_cross_samples(callset):
    """Two samples hold identical chr1 clusters; they must stay two components.

    Without `sample_id` in the component key, 89% of overlap edges joined
    different people's genomes and component counts collapsed by ~72%.
    """
    merged = merge_components(callset, min_reciprocal_overlap=0.5)
    three = merged.select(merged.n_sources >= 3)

    assert len(three) == 2, "expected one three-caller component per sample"
    samples = merged.parent.sample_idx[three.representative]
    assert len(np.unique(samples)) == 2, "the components must belong to different samples"


def test_reciprocal_overlap_is_a_real_threshold(callset):
    """The ~9% overlapping chr2 pair merges at 0.0 and must not at 0.5."""
    loose = filter_edges(callset, min_reciprocal_overlap=0.0)
    strict = filter_edges(callset, min_reciprocal_overlap=0.5)

    assert len(loose) > len(strict)
    assert len(merge_components(callset, loose)) < len(merge_components(callset, strict))
