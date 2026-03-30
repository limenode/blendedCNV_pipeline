import argparse
from pathlib import Path
from typing import Any, List, Tuple, Optional
import numpy as np
from enum import Enum
import yaml
import requests
from urllib.parse import urlparse
from urllib.request import urlretrieve
import os

class DistributionType(Enum):
    DENSITY = "density"
    CUMULATIVE = "cumulative"
    COMPLEMENTARY_CUMULATIVE = "complementary_cumulative"

class SVType(Enum):
    DEL = "DEL"
    DUP = "DUP"
    ALL = "ALL"

def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description='Process CNV files from multiple tools')
    parser.add_argument('config', type=Path, help='Path to configuration YAML file')
    args = parser.parse_args()

    # Load configuration from YAML file
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process benchmark_map to handle URLs
    if 'benchmark_map' in config and config['benchmark_map']:
        # Get project root (parent of the config file's directory or workspace root)
        project_root = Path(args.config).parent
        tmp_dir = project_root / 'tmp'
        tmp_dir.mkdir(exist_ok=True)
        
        for benchmark_name, benchmark_path in config['benchmark_map'].items():
            if isinstance(benchmark_path, str) and _is_url(benchmark_path):
                print(f"Downloading {benchmark_name} from {benchmark_path}...")
                local_path = _download_benchmark(benchmark_path, tmp_dir, benchmark_name)
                config['benchmark_map'][benchmark_name] = str(local_path)
                print(f"Saved to: {local_path}")
    
    return config

def _is_url(path: str) -> bool:
    """Check if a path is a URL."""
    try:
        result = urlparse(path)
        return result.scheme in ('http', 'https', 'ftp', 'ftps')
    except:
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
        print(f"File already exists at {local_path}, skipping download")
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

def generate_size_intervals_old(
    start: float, 
    end: float, 
    n_points: int, 
    distribution_type: DistributionType = DistributionType.DENSITY
) -> List[Tuple[float, float]]:
    """
    Generate size intervals for different distribution analyses.
    
    Creates logarithmically-spaced points and generates intervals based on the
    distribution type:
    - density: Adjacent pairs (bin intervals)
    - cumulative: Start value to each point (cumulative from beginning)
    - complementary_cumulative: Each point to end value (cumulative from end)
    
    Args:
        start: Starting value (lower bound)
        end: Ending value (upper bound)
        n_points: Number of points in logspace
        distribution_type: Type of intervals to generate
    Returns:
        List of (lower, upper) tuples representing size intervals
    
    Examples:
        >>> generate_size_intervals(1e3, 1e6, 10, "density")
        [(1000, 2154), (2154, 4642), ..., (464159, 1000000)]
        
        >>> generate_size_intervals(1e3, 1e6, 10, "cumulative")
        [(1000, 1000), (1000, 2154), ..., (1000, 1000000)]
        
        >>> generate_size_intervals(1e3, 1e6, 10, "complementary_cumulative")
        [(1000, 1000000), (2154, 1000000), ..., (1000000, 1000000)]
    """
    # Generate logarithmically-spaced points
    points = np.logspace(np.log10(start), np.log10(end), n_points)
    
    intervals = []
    
    if distribution_type == DistributionType.DENSITY:
        # Adjacent pairs: bin intervals for density distribution
        for i in range(len(points) - 1):
            intervals.append((points[i], points[i + 1]))
    
    elif distribution_type == DistributionType.CUMULATIVE:
        # Start to each point: cumulative distribution
        for point in points:
            intervals.append((start, point))
        # Remove the first interval if it is (start, start) to avoid zero-length interval
        if intervals and intervals[0][0] >= intervals[0][1]:
            intervals.pop(0)
    
    elif distribution_type == DistributionType.COMPLEMENTARY_CUMULATIVE:
        # Each point to end: complementary cumulative distribution
        for point in points:
            intervals.append((point, end))
        # Remove the last interval if it is (end, end) to avoid zero-length interval
        if intervals and intervals[-1][0] >= intervals[-1][1]:
            intervals.pop()
    
    else:
        raise ValueError(
            f"Unknown distribution_type: '{distribution_type}'. "
            f"Must be 'density', 'cumulative', or 'complementary_cumulative'"
        )
    
    return intervals

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

def get_count_from_bed_file(bed_file: str | Path) -> int:
    """Count the number of records in a BED file."""
    with open(bed_file, 'r') as f:
        return sum(1 for _ in f)

def ensure_chr_prefix(chrom: str) -> str:
    """Ensure chromosome name has 'chr' prefix."""
    if not chrom.startswith('chr'):
        return f'chr{chrom}'
    return chrom

def sanitize_svtype(svtype: Optional[str], record_id: str = "") -> str:
    """
    Sanitize SVTYPE to unified DEL/DUP classification.
    Maps all insertion types (INS, LINE1, ALU, SVA) to DUP for consistency.
    
    Args:
        svtype: SVTYPE from INFO field
        record_id: Record ID (may contain type information for CNV records)
        
    Returns:
        Sanitized type: 'DEL', 'DUP', or 'NA'
    """
    if svtype is None:
        return 'NA'
    
    svtype_upper = svtype.upper()
    
    # Handle deletions - check if any deletion pattern is in the string
    if any(pattern in svtype_upper for pattern in {'DEL', 'DELETION'}):
        return 'DEL'
    
    # Handle duplications and insertions - check if any dup/ins pattern is in the string
    if any(pattern in svtype_upper for pattern in {'DUP', 'DUPLICATION', 'INS', 'INSERTION'}):
        return 'DUP'
    
    # Handle mobile element insertions as duplications
    if any(pattern in svtype_upper for pattern in {'LINE1', 'ALU', 'SVA'}):
        return 'DUP'
    
    # Handle CNV type by checking ID field
    if 'CNV' in svtype_upper:
        record_id_upper = record_id.upper()
        if 'DEL' in record_id_upper:
            return 'DEL'
        elif 'DUP' in record_id_upper:
            return 'DUP'
    
    # Explicitly handle known SVTYPEs that we want to ignore (e.g., BND, INV)
    if any(pattern in svtype_upper for pattern in {'BND', 'INV', 'TRA', 'CTX'}):
        return 'NA'
    
    print(f"Warning: Unrecognized SVTYPE '{svtype}' in record ID '{record_id}'. Skipping.")
    exit(1)
    return 'NA'