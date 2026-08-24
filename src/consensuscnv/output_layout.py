"""Pipeline's output directory tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _slug(name: str) -> str:
    """Map a config key like ``'30x Coverage'`` to its directory name."""
    return name.replace(" ", "_")

@dataclass(frozen=True)
class OutputLayout:
    """Owns the ``output_dir`` tree. Construct once from config, pass it down."""

    root: Path

    def call_set_dir(self, origin_set: str) -> Path:
        return self.root / _slug(origin_set)

    def bed_tool_dir(self, origin_set: str, tool: str) -> Path:
        return self.call_set_dir(origin_set) / tool


    def control_dir(self, control_key: str) -> Path:
        return self.root / _slug(control_key)

    def control_bed_dir(self, control_key: str) -> Path:
        return self.control_dir(control_key) / "bed"

    @property
    def benchmark(self) -> Path:
        """Parsed benchmarks directory."""
        return self.root / "benchmark"

    def benchmark_dir(self, benchmark_key: str) -> Path:
        return self.benchmark / _slug(benchmark_key)
