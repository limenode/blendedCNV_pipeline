"""One side of a classification: flat intervals plus the columns needed to
partition, label, and write them back out.
"""

from dataclasses import dataclass, replace

import numpy as np

from consensuscnv.callsets.callset import CallSet
from consensuscnv.callsets.merging import MergedCallSet


@dataclass(frozen=True, slots=True)
class IntervalSet:
    """One side of a classification: flat intervals plus their partition columns."""

    starts: np.ndarray
    ends: np.ndarray

    chrom_idx: np.ndarray
    svtype_idx: np.ndarray
    sample_idx: np.ndarray
    source_bits: np.ndarray

    origin: CallSet
    row_index: np.ndarray

    def __len__(self) -> int:
        return len(self.starts)

    @property
    def lengths(self) -> np.ndarray:
        """Interval length in base pairs. ``ends - starts``."""
        return self.ends - self.starts

    @property
    def n_sources(self) -> np.ndarray:
        """Distinct callers behind each interval."""
        return np.bitwise_count(self.source_bits)

    @classmethod
    def from_callset(cls, callset: CallSet) -> "IntervalSet":
        return cls(
            starts=callset.starts,
            ends=callset.ends,
            chrom_idx=callset.chrom_idx,
            svtype_idx=callset.svtype_idx,
            sample_idx=callset.sample_idx,
            source_bits=callset.source_bits,
            origin=callset,
            row_index=np.arange(len(callset), dtype=np.int64),
        )

    @classmethod
    def from_merged(cls, merged: MergedCallSet) -> "IntervalSet":
        parent = merged.parent
        representative = merged.representative
        return cls(
            starts=merged.starts,
            ends=merged.ends,
            chrom_idx=parent.chrom_idx[representative],
            svtype_idx=parent.svtype_idx[representative],
            sample_idx=parent.sample_idx[representative],
            source_bits=merged.source_bits,
            origin=parent,
            row_index=representative
        )

    def select(self, rows: np.ndarray) -> "IntervalSet":
        """Get a new IntervalSet holding only `rows` - a boolean mask or integer array of row indices."""
        return replace(
            self,
            starts=self.starts[rows],
            ends=self.ends[rows],
            chrom_idx=self.chrom_idx[rows],
            svtype_idx=self.svtype_idx[rows],
            sample_idx=self.sample_idx[rows],
            source_bits=self.source_bits[rows],
            row_index=self.row_index[rows]
        )

    def filter_by_size(self, min_size: int = 0, max_size: int | None = None) -> "IntervalSet":
        """Get a new IntervalSet holding only intervals within the given size range."""
        lengths = self.lengths
        keep = lengths >= min_size
        if max_size is not None:
            keep &= lengths <= max_size
        return self.select(keep)

    def restrict_to_samples(self, sample_idx: np.ndarray) -> "IntervalSet":
        """Get a new IntervalSet holding only intervals from the given samples."""
        keep = np.isin(self.sample_idx, sample_idx)
        return self.select(keep)
