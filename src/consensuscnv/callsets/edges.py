"""Selecting which edges of a CallSet count at a given parameter point."""

from dataclasses import dataclass

import numpy as np

from consensuscnv.callsets.callset import CallSet


@dataclass(frozen=True, slots=True)
class EdgeSelection:
    a: np.ndarray
    b: np.ndarray
    min_reciprocal_overlap: float
    max_padding: int | None

    def __len__(self) -> int:
        return len(self.a)


def filter_edges(
    callset: CallSet,
    min_reciprocal_overlap: float = 0.0,
    max_padding: int | None = None,
    *,
    allow_mixed: bool = False,
) -> EdgeSelection:
    """Select edges passing reciprocal overlap and/or padding thresholds.

    `max_padding=None` means no padding at all.
    `max_padding=0` bridges exactly-touching intervals (gap distance 0).

    `max_padding` cannot exceed the `search_radius` the CallSet was built with:
    pairs further apart were never recorded, so the selection would be silently
    incomplete. Overlap edges are complete at every radius.
    """

    if min_reciprocal_overlap > 0.0 and max_padding is not None and not allow_mixed:
        raise ValueError(
            f"min_reciprocal_overlap={min_reciprocal_overlap} and max_padding={max_padding} are both set, \
            but allow_mixed is False. This drops overlapping pairs below the threshold while keeping \
            non-overlapping pairs within the padding, which is not interpretable."
        )

    if max_padding is not None and max_padding > callset.search_radius:
        raise ValueError(
            f"max_padding={max_padding} exceeds the search_radius this CallSet was built with "
            f"({callset.search_radius}). Pairs beyond that radius were never recorded, so the "
            "result would be silently incomplete. Rebuild with a wider search_radius."
        )

    ov_key = callset.ov_key

    recip_idx = np.searchsorted(ov_key, min_reciprocal_overlap, side="left")

    if max_padding is None:
        return EdgeSelection(
            a=callset.ov_a[recip_idx:],
            b=callset.ov_b[recip_idx:],
            min_reciprocal_overlap=min_reciprocal_overlap,
            max_padding=None,
        )

    padding_idx = np.searchsorted(callset.gap_key, max_padding, side="right")

    if padding_idx == 0:
        a, b = callset.ov_a[recip_idx:], callset.ov_b[recip_idx:]
    elif recip_idx == len(ov_key):
        a, b = callset.gap_a[:padding_idx], callset.gap_b[:padding_idx]
    else:
        a = np.concatenate((callset.ov_a[recip_idx:], callset.gap_a[:padding_idx]))
        b = np.concatenate((callset.ov_b[recip_idx:], callset.gap_b[:padding_idx]))

    return EdgeSelection(
        a=a, b=b, min_reciprocal_overlap=min_reciprocal_overlap, max_padding=max_padding
    )
