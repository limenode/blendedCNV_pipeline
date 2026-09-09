import argparse
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import requests
import yaml
from liftover import ChainFile

from consensuscnv.output_layout import OutputLayout


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

    # --- Optional sections (empty/None if absent) ---
    control: dict[str, str] = field(default_factory=dict)
    benchmark: dict[str, str] = field(default_factory=dict)
    liftover: dict[str, dict[str, str]] = field(default_factory=dict)
    excluded_regions_file: str | None = None
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

            control=raw.get('control', {}),
            benchmark=raw.get('benchmark', {}),
            liftover=raw.get('liftover', {}),
            excluded_regions_file=raw.get('excluded_regions_file') or None,
            sample_list_file=raw.get('sample_list_file') or None,
        )

def build_config(config_path: Path) -> PipelineConfig:
    """Load a config YAML and build a PipelineConfig.

    The chromosome domain is derived from `genome_file` in `PipelineConfig.from_raw`.
    `parse_args` wraps this for CLI use; callers can use it directly with a path.
    """
    print(f"Loading configuration from: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve benchmark URLs to local files.
    # TODO(PLAN.md Step 1): this belongs on the benchmark path, not in config
    # loading -- building a config for a consensus run should not start a download.
    if config.get('benchmark'):
        project_root = Path(config_path).parent
        tmp_dir = project_root / 'tmp'
        tmp_dir.mkdir(exist_ok=True)

        for benchmark_name, benchmark_path in config['benchmark'].items():
            if isinstance(benchmark_path, str) and _is_url(benchmark_path):
                local_path = _download_benchmark(benchmark_path, tmp_dir, benchmark_name)
                config['benchmark'][benchmark_name] = str(local_path)

    return PipelineConfig.from_raw(config)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description='Process CNV files from multiple tools')
    parser.add_argument('config', type=Path, help='Path to configuration YAML file')
    return build_config(parser.parse_args().config)

def _is_url(path: str) -> bool:
    """Check if a path is a URL."""
    try:
        result = urlparse(path)
        return result.scheme in ('http', 'https', 'ftp', 'ftps')
    except ValueError:
        return False

def _download_benchmark(url: str, tmp_dir: Path, benchmark_name: str) -> Path:
    """Download a benchmark file from a URL to the tmp directory."""
    # Extract filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)

    # If no filename in URL, use benchmark name
    if not filename:
        filename = f"{benchmark_name}.vcf.gz"

    # Create a unique filename with benchmark name prefix
    local_path = tmp_dir / f"{benchmark_name}_{filename}"

    # Check if file already exists
    if local_path.exists():
        # print(f"File already exists at {local_path}, skipping download")
        return local_path

    # Download the file (use urllib for FTP, requests for HTTP/HTTPS)
    if parsed_url.scheme in ('ftp', 'ftps'):
        # Use urllib for FTP downloads
        urlretrieve(url, local_path)
    else:
        # Use requests for HTTP/HTTPS downloads with streaming
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return local_path

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
