import argparse
from pathlib import Path
from typing import Any, List, Tuple, Callable
import numpy as np
from enum import Enum
import yaml

class DistributionType(Enum):
    DENSITY = "density"
    CUMULATIVE = "cumulative"
    COMPLEMENTARY_CUMULATIVE = "complementary_cumulative"

class SVType(Enum):
    DEL = "DEL"
    DUP = "DUP"
    ALL = "ALL"

def parse_args():
    parser = argparse.ArgumentParser(description='Process CNV files from multiple tools')
    parser.add_argument('config', type=Path, help='Path to configuration YAML file')
    args = parser.parse_args()

    # Load configuration from YAML file
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

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
    