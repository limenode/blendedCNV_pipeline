"""One side of a classification: flat intervals plus the columns needed to
partition, label, and write them back out.

`from_bed` and `from_records` are the front door: one call takes files or plain
intervals to an IntervalSet whose `origin` CallSet carries the overlap graph.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from consensuscnv.callsets.bed_io import source_strings_for
from consensuscnv.callsets.calls import PARTITION_FIELDS, calls_from_records
from consensuscnv.callsets.callset import CallSet, CallSource, collect_callsets
from consensuscnv.callsets.merging import MergedCallSet
from consensuscnv.callsets.registry import (
    CHROMOSOMES,
    SAMPLES,
    SVTYPES,
    read_genome_file,
    seed_chromosomes,
)

if TYPE_CHECKING:  # pandas costs ~300 ms to import; only `to_frame` needs it
    import pandas as pd


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

    # Names, looked up from the registries. The `_idx` columns are what the
    # package computes with; these are for reading results and choosing rows.
    @property
    def chrom_names(self) -> np.ndarray:
        return np.asarray(CHROMOSOMES.names, dtype=object)[self.chrom_idx]

    @property
    def svtype_names(self) -> np.ndarray:
        return np.asarray(SVTYPES.names, dtype=object)[self.svtype_idx]

    @property
    def sample_names(self) -> np.ndarray:
        return np.asarray(SAMPLES.names, dtype=object)[self.sample_idx]

    @property
    def source_names(self) -> list[str]:
        """The callers behind each interval, pipe-joined, as `write_merged_bed` writes them."""
        return source_strings_for(self.source_bits.tolist())

    @property
    def samples(self) -> list[str]:
        """The distinct sample names present, in registry order."""
        return [SAMPLES.names[i] for i in np.unique(self.sample_idx)]

    @classmethod
    def from_bed(
        cls,
        paths: CallSource | Iterable[CallSource],
        *,
        genome: str | Path | Iterable[str] | None = None,
        partition_by: Iterable[str] = PARTITION_FIELDS,
        search_radius: int = 0,
    ) -> IntervalSet:
        """Intervals from BED files, with their overlap graph built.

        `paths` is one path, a glob pattern, or any iterable of them -- anything
        `collect_callsets` takes, so a CallSet or an iterable of Calls works too.
        `genome` is the genome file, or the chromosome names, of the analysis: it
        seeds the chromosome registry (needed once per process, a no-op after)
        and is the order rows are sorted into. Without it the registry's
        existing order is used.
        """
        chromosome_order = None
        if genome is not None:
            chromosome_order = (
                read_genome_file(genome) if isinstance(genome, (str, Path)) else tuple(genome)
            )
            seed_chromosomes(chromosome_order)
        callset = collect_callsets(
            paths,
            chromosome_order=chromosome_order,
            partition_by=partition_by,
            search_radius=search_radius,
        )
        return cls.from_callset(callset)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping | tuple],
        *,
        genome: str | Path | Iterable[str] | None = None,
        svtype: str = "CNV",
        source: str = "user",
        sample_id: str = "sample",
        partition_by: Iterable[str] = PARTITION_FIELDS,
        search_radius: int = 0,
    ) -> IntervalSet:
        """Intervals from plain records, with their overlap graph built.

        `records` and the three field defaults are `calls_from_records`'s: each
        record is ``(chrom, start, end[, svtype[, source[, sample_id]]])`` or a
        mapping with those keys. `genome`, `partition_by` and `search_radius`
        are as for `from_bed`.
        """
        calls = calls_from_records(records, svtype=svtype, source=source, sample_id=sample_id)
        return cls.from_bed(
            calls, genome=genome, partition_by=partition_by, search_radius=search_radius
        )

    @classmethod
    def from_callset(cls, callset: CallSet) -> IntervalSet:
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
    def from_merged(cls, merged: MergedCallSet) -> IntervalSet:
        """One interval per component.

        chrom / svtype / sample come off the representative, which is only valid
        when the parent partitions on them; `source_bits` must come from the
        merged set, since the representative is one call from one caller.
        """
        return cls(
            starts=merged.starts,
            ends=merged.ends,
            chrom_idx=merged.chrom_idx,
            svtype_idx=merged.svtype_idx,
            sample_idx=merged.sample_idx,
            source_bits=merged.source_bits,
            origin=merged.parent,
            row_index=merged.representative,
        )

    def select(self, rows: np.ndarray) -> IntervalSet:
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

    def filter_by_size(self, min_size: int = 0, max_size: int | None = None) -> IntervalSet:
        """Get a new IntervalSet holding only intervals within the given size range."""
        lengths = self.lengths
        keep = lengths >= min_size
        if max_size is not None:
            keep &= lengths <= max_size
        return self.select(keep)

    def restrict_to_samples(
        self, samples: Iterable[str] | Iterable[int] | np.ndarray, *, invert: bool = False
    ) -> IntervalSet:
        """Get a new IntervalSet holding only intervals from the given samples.

        `samples` is sample names or registry ids. A name the registry has never
        seen matches nothing. `invert=True` keeps every other sample instead.
        """
        wanted = np.asarray(list(samples) if not isinstance(samples, np.ndarray) else samples)
        if wanted.dtype.kind in "US" or wanted.dtype == object:
            wanted = np.array([SAMPLES.get(str(name)) for name in wanted], dtype=np.int64)
        keep = np.isin(self.sample_idx, wanted)
        return self.select(~keep if invert else keep)

    def to_frame(self) -> pd.DataFrame:
        """The intervals as a DataFrame of names, one row per interval, in row order.

        Columns are ``chrom start end svtype source sample_id`` -- the BED
        layout -- plus ``n_sources``.
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "chrom": self.chrom_names,
                "start": self.starts,
                "end": self.ends,
                "svtype": self.svtype_names,
                "source": self.source_names,
                "sample_id": self.sample_names,
                "n_sources": self.n_sources,
            }
        )
