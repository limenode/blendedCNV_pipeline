"""Pipeline's output directory tree.

Every producer and consumer should build paths through ``OutputLayout`` instead
of concatenating strings, so the on-disk contract lives in exactly one place.

The layout computes paths and never creates directories."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import ClassVar


def _slug(name: str) -> str:
    """Map a config key like ``'30x Coverage'`` to its directory name."""
    return name.replace(" ", "_")


def _fmt(value: object) -> str:
    """Render a parameter value into a compact, filesystem-safe token.

    Floats use ``%g`` so ``0.5 -> '0.5'`` and ``0.0 -> '0'`` (no trailing zeros);
    bools collapse to ``T``/``F``. Everything else falls back to ``str``.
    """
    if isinstance(value, bool):  # bool before float: bool is an int subclass
        return "T" if value else "F"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


@dataclass(frozen=True)
class ParamSet:
    """Base for parameter bundles that render to a deterministic directory slug.

    Subclasses declare their tunable knobs as dataclass fields and set a class
    ``_prefix``. ``slug()`` walks the fields in declaration order and joins
    ``<label><value>`` tokens with underscores, e.g. ``bench_pad500_mn1_mw0_lssT``.
    Give a field a ``metadata={"label": "..."}`` to control its token name;
    otherwise the field name is used.
    """

    _prefix: ClassVar[str] = ""

    def slug(self) -> str:
        tokens = [
            f"{f.metadata.get('label', f.name)}{_fmt(getattr(self, f.name))}"
            for f in fields(self)
        ]
        body = "_".join(tokens)
        return f"{self._prefix}_{body}" if self._prefix else body


@dataclass(frozen=True)
class ConsensusParams(ParamSet):
    """Tunables for merging the query call sets into consensus components.

    ``min_weight`` is the ``min_reciprocal_overlap`` passed to
    ``callsets.merging.merge_components``. The consensus level is carried
    separately, since one merge serves all three (``min_sources`` is a post-hoc
    filter on the components, so selecting ``n_sources >= k`` off an unfiltered
    merge reproduces ``min_sources=k`` exactly).

    The field name and its ``w`` label are kept as-is because the slug they
    produce is part of the on-disk contract; renaming either changes every
    directory name.
    """

    min_weight: float = field(default=0.5, metadata={"label": "w"})


@dataclass(frozen=True)
class BenchmarkMergeParams(ParamSet):
    """Tunables for merging the benchmark truth set.

    The truth side merges on padding rather than reciprocal overlap, so
    ``padding`` maps to ``merge_components(max_padding=...)``. Note that
    ``max_padding=None`` disables padding entirely while ``max_padding=0``
    bridges exactly-touching intervals -- they are not the same setting.
    """

    _prefix: ClassVar[str] = "bench"

    padding: int = field(default=0, metadata={"label": "pad"})
    min_nodes: int = field(default=1, metadata={"label": "mn"})
    min_weight: float = field(default=0.0, metadata={"label": "mw"})
    link_same_source: bool = field(default=True, metadata={"label": "lss"})


@dataclass(frozen=True)
class ClassificationParams(ParamSet):
    """Tunables for matching query intervals against the merged truth set."""

    _prefix: ClassVar[str] = "classify"

    reciprocal_threshold: float = field(default=0.5, metadata={"label": "recip"})


@dataclass(frozen=True)
class OutputLayout:
    """Owns the ``output_dir`` tree. Construct once from config, pass it down."""

    root: Path

    # ------------------------------------------------------------------ #
    # Per experimental call set (e.g. '30x Coverage')
    # ------------------------------------------------------------------ #
    def call_set_dir(self, origin_set: str) -> Path:
        return self.root / _slug(origin_set)

    def bed_tool_dir(self, origin_set: str, tool: str) -> Path:
        return self.call_set_dir(origin_set) / tool

    # ------------------------------------------------------------------ #
    # Per control call set (e.g. 'SNP Array')
    # ------------------------------------------------------------------ #
    def control_dir(self, control_key: str) -> Path:
        return self.root / _slug(control_key)

    def control_bed_dir(self, control_key: str) -> Path:
        return self.control_dir(control_key) / "bed"

    # ------------------------------------------------------------------ #
    # Per benchmark
    # ------------------------------------------------------------------ #
    @property
    def benchmark(self) -> Path:
        """Parsed benchmarks directory."""
        return self.root / "benchmark"

    def benchmark_dir(self, benchmark_key: str) -> Path:
        return self.benchmark / _slug(benchmark_key)
