"""Single source of truth for the pipeline's output directory tree.

Every producer and consumer should build paths through ``OutputLayout`` instead
of concatenating strings, so the on-disk contract lives in exactly one place.

The layout is pure: it computes paths and never creates directories — callers
keep doing their own ``mkdir``. This keeps it trivial to read and test.

Seam with the shell scripts: the consensus scripts receive a ``consensus_<k>of3``
directory as an argument and themselves create the ``intersections/`` and
``unions/`` subdirectories (and the per-sample file names). ``OutputLayout``
therefore owns the tree down to :meth:`consensus_dir`; :meth:`consensus_rep_dir`
mirrors the subdirectories the scripts produce so Python consumers can find them.

This is the Phase-1 layout: every path is identical to the current build. Change
the tree by editing method bodies here, not the call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CONSENSUS_LEVELS = (1, 2, 3)
REPRESENTATIONS = ("intersections", "unions")


def _slug(name: str) -> str:
    """Map a config key like ``'30x Coverage'`` to its directory name."""
    return name.replace(" ", "_")


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
        """Parsed (merged) gold-standard benchmark BEDs."""
        return self.root / "benchmark"

    def log(self, filename: str) -> Path:
        return self.logs / filename

    def figure_group(self, name: str) -> Path:
        return self.figures / name

    # ------------------------------------------------------------------ #
    # Global summary files (currently emitted at the root, beside logs/)
    # ------------------------------------------------------------------ #
    @property
    def counts_summary(self) -> Path:
        return self.root / "analysis_counts_summary.json"

    @property
    def counts_summary_all(self) -> Path:
        return self.root / "analysis_counts_summary_all.json"

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

    def bed_dir(self, set_key: str, tool: str) -> Path:
        return self.set_dir(set_key) / "bed" / tool

    def consensus_dir(self, set_key: str, level: int) -> Path:
        """``consensus_<level>of3`` — the directory handed to the consensus scripts."""
        return self.set_dir(set_key) / f"consensus_{level}of3"

    def consensus_rep_dir(self, set_key: str, level: int, representation: str) -> Path:
        """``intersections/`` or ``unions/`` subdir created by the consensus scripts."""
        return self.consensus_dir(set_key, level) / representation

    def classification_root(self, set_key: str) -> Path:
        return self.set_dir(set_key) / "binary_classification"

    def classification_dir(self, set_key: str, call_set: str) -> Path:
        """``binary_classification/<call_set>`` for one call set (a tool or a consensus set)."""
        return self.classification_root(set_key) / call_set

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
    # Naming conventions (centralized so the contract stays in one file)
    # ------------------------------------------------------------------ #
    @staticmethod
    def consensus_call_set(level: int, representation: str) -> str:
        """Name of a consensus call set, e.g. ``'consensus_2of3_intersections'``.

        Used both as a ``binary_classification`` subdirectory and as an analysis key.
        """
        return f"consensus_{level}of3_{representation}"

    @staticmethod
    def sample_bed(sample: str, svtype: str) -> str:
        """Per-sample BED filename, e.g. ``'HG002.DEL.bed'``."""
        return f"{sample}.{svtype}.bed"

    @staticmethod
    def classification_bed(sample: str, svtype: str, label: str) -> str:
        """TP/FP/FN BED filename, e.g. ``'HG002.DEL.TP.bed'`` (``label`` is TP/FP/FN)."""
        return f"{sample}.{svtype}.{label}.bed"
