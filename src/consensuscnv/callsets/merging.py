"""Collapsing connected components of a filtered CallSet into merged calls.

`MergedCallSet` is a *view over its parent CallSet*: `representative` and `labels`
are indices into it, and chrom / svtype / sample_id are read back off the
representative rather than stored, since they are uniform within a component.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csgraph

from consensuscnv.callsets.callset import CallSet
from consensuscnv.callsets.edges import EdgeSelection, filter_edges


@dataclass(frozen=True, slots=True)
class MergedCallSet:
    representative: np.ndarray  # index of one member call, per component
    starts: np.ndarray
    ends: np.ndarray
    source_bits: np.ndarray
    n_calls: np.ndarray
    component_id: np.ndarray  # original label; survives min_* filtering
    labels: np.ndarray  # node -> label, for every node of the parent CallSet

    def __len__(self) -> int:
        return len(self.starts)

    @property
    def n_sources(self) -> np.ndarray:
        return np.bitwise_count(self.source_bits)  # Count the number of unique sources for each merged call


def merge_components(
    callset: CallSet,
    selection: EdgeSelection | None = None,
    *,
    min_reciprocal_overlap: float = 0.0,
    max_padding: int | None = None,
    min_calls: int = 1,
    min_sources: int = 1,
) -> MergedCallSet:
    """Merge each connected component of the selected edges into a single call.

    `min_calls` counts member calls; `min_sources` counts *distinct* callers.
    """
    if selection is None:
        selection = filter_edges(
            callset,
            min_reciprocal_overlap=min_reciprocal_overlap,
            max_padding=max_padding,
        )

    n = len(callset.calls)
    graph = coo_matrix((np.ones(len(selection), np.int8), (selection.a, selection.b)), shape=(n, n))

    n_components, labels = csgraph.connected_components(graph, directed=False, return_labels=True)

    # One reduction per column. `ufunc.at` scatters element i into slot labels[i],
    # combining collisions with the ufunc rather than overwriting.
    starts = np.full(n_components, np.iinfo(np.int64).max, dtype=np.int64)
    ends = np.zeros(n_components, dtype=np.int64)
    source_bits = np.zeros(n_components, dtype=np.int64)
    representative = np.full(n_components, n, dtype=np.int64)

    np.minimum.at(starts, labels, callset.starts)
    np.maximum.at(ends, labels, callset.ends)
    np.bitwise_or.at(source_bits, labels, callset.source_bits)
    np.minimum.at(representative, labels, np.arange(n, dtype=np.int64))
    n_calls = np.bincount(labels, minlength=n_components)

    component_id = np.arange(n_components, dtype=np.int64)
    if min_calls > 1 or min_sources > 1:
        keep = ((n_calls >= min_calls) & (np.bitwise_count(source_bits) >= min_sources))
        representative, starts, ends, source_bits, n_calls, component_id = (
            x[keep] for x in (representative, starts, ends, source_bits, n_calls, component_id)
        )

    return MergedCallSet(
        representative=representative,
        starts=starts,
        ends=ends,
        source_bits=source_bits,
        n_calls=n_calls,
        component_id=component_id,
        labels=labels,
    )
