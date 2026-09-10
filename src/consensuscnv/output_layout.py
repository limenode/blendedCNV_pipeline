"""Pipeline's output directory tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RESERVED_NAMES = frozenset({"benchmark", "consensus", "downloads", "evaluation"})


def slug(name: str) -> str:
    """Map a config key like ``'30x Coverage'`` to its directory name."""
    return name.replace(" ", "_")


def overlap_slug(reciprocal_overlap: float) -> str:
    """Directory name for one consensus reciprocal-overlap setting.

    Returns with overlap truncated to two decimal places, e.g. ``overlap_0.50``.
    """
    return f"overlap_{reciprocal_overlap:.2f}"


@dataclass(frozen=True)
class OutputLayout:
    """Owns the ``output_dir`` tree. Construct once from config, pass it down."""

    root: Path

    def call_set_dir(self, origin_set: str) -> Path:
        return self.root / slug(origin_set)

    def bed_tool_dir(self, origin_set: str, tool: str) -> Path:
        return self.call_set_dir(origin_set) / tool

    def control_dir(self, control_key: str) -> Path:
        return self.root / slug(control_key)

    def control_bed_dir(self, control_key: str) -> Path:
        return self.control_dir(control_key) / "bed"

    @property
    def consensus(self) -> Path:
        """Root of the consensus output tree."""
        return self.root / "consensus"

    def consensus_set_dir(self, origin_set: str) -> Path:
        return self.consensus / slug(origin_set)

    def consensus_overlap_dir(self, origin_set: str, reciprocal_overlap: float) -> Path:
        return self.consensus_set_dir(origin_set) / overlap_slug(reciprocal_overlap)

    def consensus_bed(
        self, origin_set: str, reciprocal_overlap: float, level: int, n_sources: int
    ) -> Path:
        """``consensus/<set>/overlap_0.50/2of3.bed``."""
        return self.consensus_overlap_dir(origin_set, reciprocal_overlap) / f"{level}of{n_sources}.bed"

    def consensus_sample_bed(
        self,
        origin_set: str,
        reciprocal_overlap: float,
        level: int,
        n_sources: int,
        sample_id: str,
    ) -> Path:
        """``consensus/<set>/overlap_0.50/2of3/HG00096.bed`` (``--per-sample``)."""
        directory = self.consensus_overlap_dir(origin_set, reciprocal_overlap)
        return directory / f"{level}of{n_sources}" / f"{sample_id}.bed"

    @property
    def evaluation(self) -> Path:
        """Metrics tables and, on request, the TP / FP / FN row files."""
        return self.root / "evaluation"

    @property
    def downloads(self) -> Path:
        """Cache for benchmark sources fetched from a URL."""
        return self.root / "downloads"

    @property
    def benchmark(self) -> Path:
        """Parsed benchmarks directory."""
        return self.root / "benchmark"

    def benchmark_dir(self, benchmark_key: str) -> Path:
        return self.benchmark / slug(benchmark_key)
