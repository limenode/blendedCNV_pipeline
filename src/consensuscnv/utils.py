import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import yaml
from liftover import ChainFile

from consensuscnv.output_layout import RESERVED_NAMES, OutputLayout, overlap_slug, slug


class DistributionType(Enum):
    DENSITY = "density"
    CUMULATIVE = "cumulative"
    COMPLEMENTARY_CUMULATIVE = "complementary_cumulative"

class SVType(Enum):
    DEL = "DEL"
    DUP = "DUP"
    ALL = "ALL"

class LiftoverStatus(Enum):
    """Outcome of lifting one interval to another genome build."""
    OK = "ok"                    # lifted successfully
    UNMAPPED = "unmapped"        # an endpoint failed to map (unknown chrom / empty result)
    SIZE_CHANGE = "size_change"  # length drifted past the allowed threshold

def read_genome_file(path: Path) -> tuple[str, ...]:
    """Ordered, de-duplicated chromosome names from a genome/faidx-style file.

    Reads column 0 of each line. Blank lines and lines whose first non-whitespace
    character is ``#`` are skipped, so commenting a chromosome out of the genome
    file is the supported way to drop it from the analysis.
    """
    names: dict[str, None] = {}  # insertion-ordered, and de-duplicates
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names[line.split()[0]] = None

    if not names:
        raise ValueError(f"No chromosomes found in genome file: {path}")
    return tuple(names)


@dataclass(frozen=True)
class ConsensusParams:
    """Parameters of the consensus merge, from the config's ``consensus:`` block.

    One consensus run is produced per entry in `reciprocal_overlaps`, each writing
    to its own directory. `0.0` reciprocal overlap means "at least 1 bp overlap"

    Every consensus level from 1 up to the number of tools configured for a call
    set is written.
    """

    reciprocal_overlaps: tuple[float, ...] = (0.5,)
    min_size: int = 1_000

    @classmethod
    def from_raw(cls, raw: dict | None) -> "ConsensusParams":
        raw = raw or {}
        overlaps = raw.get("reciprocal_overlap", cls.reciprocal_overlaps)
        if isinstance(overlaps, (int, float)):  # a bare scalar is a list of one
            overlaps = [overlaps]
        return cls(
            reciprocal_overlaps=tuple(float(value) for value in overlaps),
            min_size=int(raw.get("min_size", cls.min_size)),
        )

    def problems(self) -> list[str]:
        """Propagates errors with input parameters, for the config's error report."""
        found = []
        if not self.reciprocal_overlaps:
            found.append("consensus.reciprocal_overlap is empty; give at least one value")
        for value in self.reciprocal_overlaps:
            if not 0.0 <= value <= 1.0:
                found.append(f"consensus.reciprocal_overlap has {value}, outside [0.0, 1.0]")

        # Each overlap names a directory, so the set has to be injective under the slug.
        slugs: dict[str, float] = {}
        for value in self.reciprocal_overlaps:
            slug = overlap_slug(value)
            if slug in slugs:
                found.append(
                    f"consensus.reciprocal_overlap has {slugs[slug]} and {value}, which "
                    f"both name the directory {slug!r}"
                )
            slugs[slug] = value

        if self.min_size < 0:
            found.append(f"consensus.min_size is {self.min_size}; must be >= 0")
        return found


@dataclass(frozen=True)
class PipelineConfig:
    """Parsed, validated pipeline configuration. Built once in ``build_config()``."""

    # --- Required ---
    experimental: dict[str, dict[str, str]]                # call_set -> {tool_label: glob_pattern}
    output_dir: Path
    genome_file: Path
    layout: OutputLayout              # derived from output_dir
    # The analysis domain, ordered, derived from `genome_file`. Parsers drop any
    # record on a contig outside it; `build_callset` sorts into this order.
    chromosomes: tuple[str, ...]
    consensus: ConsensusParams = field(default_factory=ConsensusParams)

    # --- Optional sections (empty/None if absent) ---
    control: dict[str, str] = field(default_factory=dict)
    benchmark: dict[str, str] = field(default_factory=dict)  # label -> local path or URL
    liftover: dict[str, dict[str, str]] = field(default_factory=dict)
    excluded_regions_file: str | None = None
    # Fraction of a call that may lie inside an excluded region before the call is
    # dropped whole. 0.0 drops on any overlap at all.
    max_excluded_fraction: float = 0.01
    sample_list_file: str | None = None   # newline-separated allowlist; None keeps all samples

    @classmethod
    def from_raw(cls, raw: dict) -> "PipelineConfig":
        output_dir = Path(raw['output_dir'])
        genome_file = Path(raw['genome_file'])
        return cls(
            experimental=raw.get('experimental', {}),
            output_dir=output_dir,
            genome_file=genome_file,
            layout=OutputLayout(output_dir),
            chromosomes=read_genome_file(genome_file),
            consensus=ConsensusParams.from_raw(raw.get('consensus')),

            control=raw.get('control', {}),
            benchmark=raw.get('benchmark', {}),
            liftover=raw.get('liftover', {}),
            excluded_regions_file=raw.get('excluded_regions_file') or None,
            max_excluded_fraction=float(
                raw.get('max_excluded_fraction', cls.max_excluded_fraction)
            ),
            sample_list_file=raw.get('sample_list_file') or None,
        )

    @property
    def liftover_keys(self) -> frozenset[str]:
        """Every name that a `liftover:` entry may key.

        A key names the dataset whose files are not in the target build: an
        `experimental` call set, a tool label inside one, a `control`, or a
        `benchmark`. See `liftover_for` for how a key is resolved.
        """
        tools = {tool for tools in self.experimental.values() for tool in tools}
        return frozenset(
            set(self.experimental) | tools | set(self.control) | set(self.benchmark)
        )

    def liftover_for(self, *names: str) -> dict[str, str] | None:
        """The `liftover:` spec for a dataset, or None if none was requested.

        `names` are the names that could describe the dataset. For an experimental
        file that is `(tool_label, call_set)`: naming the tool lifts that caller's
        output wherever it appears, naming the call set lifts everything in it, and
        naming both lets the tool take precedence. Controls and benchmarks have only
        their own name.
        """
        for name in names:
            spec = self.liftover.get(name)
            if spec:
                return spec
        return None

    def __post_init__(self) -> None:
        """Reject a structurally invalid config, reporting every fault."""
        problems: list[str] = []

        if not self.experimental:
            problems.append("experimental is empty; give at least one call set")
        for call_set, tools in self.experimental.items():
            if not tools:
                problems.append(f"experimental[{call_set!r}] has no tools")

        # Call sets and controls both become directories directly under output_dir.
        for section in ("experimental", "control"):
            for name in getattr(self, section):
                if slug(name).lower() in RESERVED_NAMES:
                    problems.append(
                        f"{section}[{name!r}] collides with a directory the pipeline "
                        f"owns; {sorted(RESERVED_NAMES)} are reserved"
                    )
        shared = set(self.experimental) & set(self.control)
        if shared:
            problems.append(
                f"{sorted(shared)} appear in both experimental and control, and would "
                "write to the same directory"
            )

        valid = self.liftover_keys
        for key in self.liftover:
            if key not in valid:
                problems.append(
                    f"liftover[{key!r}] names no dataset; expected a tool label, a "
                    f"control, or a benchmark -- one of {sorted(valid)}"
                )
        for key, spec in self.liftover.items():
            missing = {"from", "to"} - set(spec or {})
            if missing:
                problems.append(f"liftover[{key!r}] is missing {sorted(missing)}")

        if not 0.0 <= self.max_excluded_fraction <= 1.0:
            problems.append(
                f"max_excluded_fraction is {self.max_excluded_fraction}, outside [0.0, 1.0]"
            )

        problems += self.consensus.problems()

        if problems:
            raise ValueError(
                "Invalid configuration:\n  - " + "\n  - ".join(problems)
            )


def build_config(config_path: Path) -> PipelineConfig:
    """Load a config YAML and build a PipelineConfig."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    parsed = PipelineConfig.from_raw(config)

    # Filesystem checks
    try:
        parsed.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(f"output_dir {parsed.output_dir} is not writable: {error}") from error

    return parsed


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description='Process CNV files from multiple tools')
    parser.add_argument('config', type=Path, help='Path to configuration YAML file')
    return build_config(parser.parse_args().config)

# Define metric functions
def precision(tp: int, fp: int, fn: int) -> float:
    """Calculate precision: TP / (TP + FP)"""
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp: int, fp: int, fn: int) -> float:
    """Calculate recall/sensitivity: TP / (TP + FN)"""
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f_beta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    """Calculate F-beta score: (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)"""
    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    beta_squared = beta ** 2
    return ((1 + beta_squared) * p * r) / (beta_squared * p + r) if (beta_squared * p + r) > 0 else 0

def f1_score(tp: int, fp: int, fn: int) -> float:
    """Calculate F1 score: 2 * (precision * recall) / (precision + recall)"""
    return f_beta_score(tp, fp, fn, beta=1.0)

def f0_5_score(tp: int, fp: int, fn: int) -> float:
    """Calculate F0.5 score: (1 + 0.5^2) * (precision * recall) / (0.5^2 * precision + recall)"""
    return f_beta_score(tp, fp, fn, beta=0.5)

def f2_score(tp: int, fp: int, fn: int) -> float:
    """Calculate F2 score: (1 + 2^2) * (precision * recall) / (2^2 * precision + recall)"""
    return f_beta_score(tp, fp, fn, beta=2.0)

def generate_size_intervals(
    start: float,
    end: float,
    n_points: int,
) -> list[tuple[float, float]]:
    """
    Generate size intervals for different distribution analyses.
    """
    points = np.logspace(np.log10(start), np.log10(end), n_points)
    intervals = []
    for i in range(len(points) - 1):
        intervals.append((points[i], points[i + 1]))

    return intervals

def ensure_chr_prefix(chrom: str) -> str:
    """Ensure chromosome name has 'chr' prefix."""
    if not chrom.startswith('chr'):
        return f'chr{chrom}'
    return chrom

def lift_interval(
    lifter: ChainFile,
    chrom: str,
    start: int,
    end: int,
    size_change_threshold: float = 0.10,
) -> tuple[LiftoverStatus, tuple[int, int] | None]:
    """Lift a (start, end) interval to another genome build.

    Returns a `(status, coords)` pair:
      - `(LiftoverStatus.OK, (start, end))`     -- lifted successfully
      - `(LiftoverStatus.UNMAPPED, None)`       -- an endpoint failed to map
                                                   (unknown chromosome or empty result)
      - `(LiftoverStatus.SIZE_CHANGE, None)`    -- length changed by more than
                                                   `size_change_threshold` (default 10%)

    Callers drop the record on any non-OK status and can attribute the drop to
    its reason. This is the shared per-record liftover used by both the VCF and
    PennCNV parsers; build the `lifter` once with
    `liftover.get_lifter(from_build, to_build)` and reuse it across records.
    """
    old_size = end - start

    try:
        new_start = lifter[chrom][start]
        new_end = lifter[chrom][end]
    except (KeyError, IndexError):
        return LiftoverStatus.UNMAPPED, None

    if not new_start or not new_end:
        return LiftoverStatus.UNMAPPED, None

    new_start, new_end = new_start[0][1], new_end[0][1]
    if old_size and abs((new_end - new_start) - old_size) / old_size > size_change_threshold:
        return LiftoverStatus.SIZE_CHANGE, None
    return LiftoverStatus.OK, (new_start, new_end)

def sanitize_svtype(svtype: str | None, record_id: str = "") -> str:
    """Sanitize SVTYPE to DEL, DUP, or NA."""
    if svtype is None:
        return 'NA'

    svtype = svtype.upper()

    if svtype in {'DEL', 'DELETION'}:
        return "DEL"
    elif svtype in {'DUP', 'DUPLICATION', 'INS', 'INSERTION', 'LINE1', 'ALU', 'SVA'}:
        return "DUP"

    # Handle CNV type by checking ID field
    if 'CNV' in svtype:
        record_id_upper = record_id.upper()
        if 'DEL' in record_id_upper:
            return 'DEL'
        elif 'DUP' in record_id_upper:
            return 'DUP'

    return 'NA'
