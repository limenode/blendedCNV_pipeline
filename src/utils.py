import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from enum import Enum
import yaml
import requests
from urllib.parse import urlparse
from urllib.request import urlretrieve
import os

from liftover import ChainFile

from output_layout import OutputLayout

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

@dataclass(frozen=True)
class PipelineConfig:
    """Parsed, validated pipeline configuration. Built once in ``parse_args()``."""

    # --- Required ---
    experimental: dict                # set_key -> {tool_label: glob_pattern}
    output_dir: Path
    genome_file: str                  # passed to shell scripts as a path string
    layout: OutputLayout              # derived from output_dir

    # --- Optional sections (empty/None if absent) ---
    control: dict = field(default_factory=dict)
    benchmark: dict = field(default_factory=dict)
    liftover: dict = field(default_factory=dict)
    valid_chromosomes: set = field(default_factory=set)
    chromosome_order: List[str] = field(default_factory=list)
    excluded_regions_file: Optional[str] = None
    analysis_plots_config: Optional[str] = None

    # --- Thresholds ---
    consensus_reciprocal_threshold: float = 0.5
    matching_reciprocal_threshold: float = 0.5

    # --- Phase gating (resolved from CLI flags in parse_args) ---
    do_processing: bool = True
    do_computation: bool = False
    do_analysis: bool = False

    @classmethod
    def from_raw(cls, raw: dict, *, do_processing: bool,
                 do_computation: bool, do_analysis: bool) -> "PipelineConfig":
        output_dir = Path(raw['output_dir'])
        return cls(
            experimental=raw.get('experimental', {}),
            output_dir=output_dir,
            genome_file=raw['genome_file'],
            layout=OutputLayout(output_dir),
            control=raw.get('control', {}),
            benchmark=raw.get('benchmark', {}),
            liftover=raw.get('liftover', {}),
            valid_chromosomes=raw.get('valid_chromosomes', set()),
            chromosome_order=raw.get('chromosome_order', []),
            excluded_regions_file=raw.get('excluded_regions_file') or None,
            analysis_plots_config=raw.get('analysis_plots_config'),
            consensus_reciprocal_threshold=raw.get('consensus_reciprocal_threshold', 0.5),
            matching_reciprocal_threshold=raw.get('matching_reciprocal_threshold', 0.5),
            do_processing=do_processing,
            do_computation=do_computation,
            do_analysis=do_analysis,
        )

def build_config(config_path: Path, *, do_processing: bool,
                 do_computation: bool, do_analysis: bool) -> PipelineConfig:
    """Load a config YAML and build a PipelineConfig.

    Parses the YAML at `config_path`, resolves benchmark URLs and valid
    chromosomes, and applies the caller-supplied phase flags. `parse_args` wraps
    this for CLI use; tests can call it directly with an explicit path.
    """
    # Load configuration from YAML file
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Process benchmark to handle URLs
    if 'benchmark' in config and config['benchmark']:
        # Get project root (parent of the config file's directory or workspace root)
        project_root = Path(config_path).parent
        tmp_dir = project_root / 'tmp'
        tmp_dir.mkdir(exist_ok=True)

        for benchmark_name, benchmark_path in config['benchmark'].items():
            if isinstance(benchmark_path, str) and _is_url(benchmark_path):
                # print(f"Downloading {benchmark_name} from {benchmark_path}...")
                local_path = _download_benchmark(benchmark_path, tmp_dir, benchmark_name)
                config['benchmark'][benchmark_name] = str(local_path)
                # print(f"Saved to: {local_path}")

    # Read genome.txt and store set of valid chromosomes
    # Get first column of genome file as set of valid chromosomes
    if 'genome_file' in config and config['genome_file']:
        genome_file = Path(config['genome_file'])
        if genome_file.exists():
            with open(genome_file, 'r') as f:
                ordered_chromosomes = [line.split()[0] for line in f if line.strip()]
            config['valid_chromosomes'] = set(ordered_chromosomes)
            config['chromosome_order'] = ordered_chromosomes
            # print(f"Loaded {len(ordered_chromosomes)} valid chromosomes from {genome_file}")
        else:
            print(f"Warning: Genome file {genome_file} not found. Chromosome validation will be skipped.")

    return PipelineConfig.from_raw(
        config,
        do_processing=do_processing,
        do_computation=do_computation,
        do_analysis=do_analysis,
    )


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description='Process CNV files from multiple tools')
    parser.add_argument('config', type=Path, help='Path to configuration YAML file')
    parser.add_argument('--run-benchmark', action='store_true', help='Whether to run benchmarking after processing')
    parser.add_argument('--only-process', action='store_true', help='Only run the processing pipeline without computation or analysis')
    parser.add_argument('--only-compute', action='store_true', help='Only run the computation pipeline without processing or analysis')
    parser.add_argument('--only-analyze', action='store_true', help='Only run the analysis pipeline without processing or computation')
    args = parser.parse_args()

    do_processing  = not (args.only_compute or args.only_analyze)
    do_computation = args.run_benchmark and not (args.only_process or args.only_analyze)
    do_analysis    = args.run_benchmark and not (args.only_process or args.only_compute)

    return build_config(
        args.config,
        do_processing=do_processing,
        do_computation=do_computation,
        do_analysis=do_analysis,
    )

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
) -> List[Tuple[float, float]]:
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
) -> Tuple[LiftoverStatus, Optional[Tuple[int, int]]]:
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

def sanitize_svtype(svtype: Optional[str], record_id: str = "") -> str:
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