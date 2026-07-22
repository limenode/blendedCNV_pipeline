"""Pipeline's output directory tree.

Every producer and consumer should build paths through ``OutputLayout`` instead
of concatenating strings, so the on-disk contract lives in exactly one place.

The layout computes paths and never creates directories.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import ClassVar

CONSENSUS_LEVELS = (1, 2, 3)
REPRESENTATIONS = ("intersections", "unions")


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
    otherwise the field name is used. Because rendering is derived purely from
    the fields, two equal param objects always map to the same path.
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
    """Tunables for ``compute_consensus_from_beds`` (the consensus level is
    carried separately, since one call emits all three levels)."""

    min_weight: float = field(default=0.5, metadata={"label": "w"})


@dataclass(frozen=True)
class BenchmarkMergeParams(ParamSet):
    """Tunables for merging the benchmark truth set (``merge_benchmarks``)."""

    _prefix: ClassVar[str] = "bench"

    padding: int = field(default=0, metadata={"label": "pad"})
    min_nodes: int = field(default=1, metadata={"label": "mn"})
    min_weight: float = field(default=0.0, metadata={"label": "mw"})
    link_same_source: bool = field(default=True, metadata={"label": "lss"})


@dataclass(frozen=True)
class ClassificationParams(ParamSet):
    """Tunables for node-level binary classification (``classify_calls``)."""

    _prefix: ClassVar[str] = "classify"

    reciprocal_threshold: float = field(default=0.5, metadata={"label": "recip"})


@dataclass(frozen=True)
class OutputLayout:
    """Owns the ``output_dir`` tree. Construct once from config, pass it down."""

    root: Path

    # ------------------------------------------------------------------ #
    # Global artifact directories
    # ------------------------------------------------------------------ #
    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def benchmark(self) -> Path:
        """Parsed benchmarks directory."""
        return self.root / "benchmark"

    def log(self, filename: str) -> Path:
        return self.logs / filename

    def figure_group(self, name: str) -> Path:
        return self.figures / name

    # ------------------------------------------------------------------ #
    # Named figure groups (as currently emitted under figures/)
    # ------------------------------------------------------------------ #
    @property
    def venn_figures(self) -> Path:
        return self.figure_group("venn_diagrams")

    @property
    def size_figures(self) -> Path:
        return self.figure_group("size_distributions")

    @property
    def caller_source_figures(self) -> Path:
        return self.figure_group("caller_source_distribution")

    @property
    def stat_dist_all_figures(self) -> Path:
        return self.figure_group("statistical_distributions_all_only")

    @property
    def stat_dist_split_figures(self) -> Path:
        return self.figure_group("statistical_distributions_split_by_svtype")

    # ------------------------------------------------------------------ #
    # Per input set (e.g. '30x Coverage')
    # ------------------------------------------------------------------ #
    def set_dir(self, set_key: str) -> Path:
        return self.root / _slug(set_key)

    def bed_dir(self, set_key: str) -> Path:
        return self.set_dir(set_key) / "bed"

    def bed_tool_dir(self, set_key: str, tool: str) -> Path:
        return self.set_dir(set_key) / "bed" / tool

    def consensus_dir(
        self, set_key: str, level: int, params: ConsensusParams | None = None
    ) -> Path:
        """``consensus_<level>`` — the directory the consensus BEDs are written to.

        When ``params`` is given, its slug is appended so runs with different
        consensus tunables land in sibling directories instead of overwriting
        each other (e.g. ``consensus_2/w0.5``)."""
        base = self.set_dir(set_key) / f"consensus_{level}"
        return base / params.slug() if params is not None else base

    def consensus_rep_dir(self, set_key: str, level: int, representation: str) -> Path:
        """``intersections/`` or ``unions/`` subdir created by the consensus scripts."""
        return self.consensus_dir(set_key, level) / representation

    def classification_root(self, benchmark_subset: str) -> Path:
        """Legacy single-level root, still used by ``analysis_driver``.

        Superseded by :meth:`classification_setting_dir` for the parameterized
        4-level tree; kept until the analysis reader migrates."""
        return self.root / "binary_classification" / _slug(benchmark_subset)

    def classification_setting_dir(
        self,
        benchmark_params: BenchmarkMergeParams,
        classification_params: ClassificationParams,
    ) -> Path:
        """``binary_classification/<bench slug>/<classify slug>`` — the subtree
        holding every input set classified under one benchmark+matching setting."""
        return (
            self.root
            / "binary_classification"
            / benchmark_params.slug()
            / classification_params.slug()
        )

    def classification_dir(
        self,
        input_set_key: str,
        call_set: str,
        *,
        benchmark_params: BenchmarkMergeParams,
        classification_params: ClassificationParams,
    ) -> Path:
        """``binary_classification/<bench>/<classify>/<input_set>/<call_set>``.

        The four levels are: benchmark-merge params, classification params, the
        input set (e.g. ``30x_Coverage``), and the call set (e.g. a caller name
        or ``consensus_2of3_w0.5``)."""
        return (
            self.classification_setting_dir(benchmark_params, classification_params)
            / _slug(input_set_key)
            / _slug(call_set)
        )

    # ------------------------------------------------------------------ #
    # Per control set (e.g. 'SNP Array')
    # ------------------------------------------------------------------ #
    def control_dir(self, control_key: str) -> Path:
        return self.root / _slug(control_key)

    def control_bed_dir(self, control_key: str) -> Path:
        return self.control_dir(control_key) / "bed"

    def control_classification_dir(self, control_key: str) -> Path:
        return self.control_dir(control_key) / "binary_classification"
    
    # ------------------------------------------------------------------ #
    # Per benchmark
    # ------------------------------------------------------------------ #
    def benchmark_dir(self, benchmark_key: str) -> Path:
        return self.benchmark / _slug(benchmark_key)

    def benchmark_merge_dir(self, params: BenchmarkMergeParams) -> Path:
        """``benchmark/merged/<bench slug>`` — merged truth set for one setting."""
        return self.benchmark / "merged" / params.slug()
    

    # ------------------------------------------------------------------ #
    # Naming conventions (centralized so the contract stays in one file)
    # ------------------------------------------------------------------ #
    @staticmethod
    def consensus_call_set(level: int, representation: str) -> str:
        """Name of a consensus call set, e.g. ``'consensus_2of3_intersections'``.

        Used both as a ``binary_classification`` subdirectory and as an analysis key.
        """
        return f"consensus_{level}of3_{representation}"

    @staticmethod
    def consensus_call_set_slug(level: int, params: ConsensusParams) -> str:
        """Call-set name for a consensus level under given params.

        Used as the ``call_set`` leaf of :meth:`classification_dir`, e.g.
        ``'consensus_2of3_w0.5'``."""
        return f"consensus_{level}of3_{params.slug()}"

    @staticmethod
    def sample_bed(sample: str, svtype: str) -> str:
        """Per-sample BED filename, e.g. ``'HG002.DEL.bed'``."""
        return f"{sample}.{svtype}.bed"

    @staticmethod
    def classification_bed(sample: str, svtype: str | None, label: str) -> str:
        """TP/FP/FN BED filename, e.g. ``'HG002.DEL.TP.bed'`` (``label`` is TP/FP/FN)."""
        if svtype is None:
            return f"{sample}.{label}.bed"
        
        return f"{sample}.{svtype}.{label}.bed"
