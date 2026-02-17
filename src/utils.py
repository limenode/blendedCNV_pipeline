import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Process CNV files from multiple tools')
    parser.add_argument('config', type=Path, help='Path to configuration YAML file')
    return parser.parse_args()

def generate_size_intervals(
    start: float, 
    end: float, 
    n_points: int, 
    distribution_type: str = "histogram"
) -> List[Tuple[float, float]]:
    """
    Generate size intervals for different distribution analyses.
    
    Creates logarithmically-spaced points and generates intervals based on the
    distribution type:
    - histogram: Adjacent pairs (bin intervals)
    - cumulative: Start value to each point (cumulative from beginning)
    - complementary_cumulative: Each point to end value (cumulative from end)
    
    Args:
        start: Starting value (lower bound)
        end: Ending value (upper bound)
        n_points: Number of points in logspace
        distribution_type: Type of intervals to generate
            - "histogram": [(p0, p1), (p1, p2), ..., (pn-1, pn)]
            - "cumulative": [(start, p0), (start, p1), ..., (start, pn)]
            - "complementary_cumulative": [(p0, end), (p1, end), ..., (pn, end)]
    
    Returns:
        List of (lower, upper) tuples representing size intervals
    
    Examples:
        >>> generate_size_intervals(1e3, 1e6, 10, "histogram")
        [(1000, 2154), (2154, 4642), ..., (464159, 1000000)]
        
        >>> generate_size_intervals(1e3, 1e6, 10, "cumulative")
        [(1000, 1000), (1000, 2154), ..., (1000, 1000000)]
        
        >>> generate_size_intervals(1e3, 1e6, 10, "complementary_cumulative")
        [(1000, 1000000), (2154, 1000000), ..., (1000000, 1000000)]
    """
    # Generate logarithmically-spaced points
    points = np.logspace(np.log10(start), np.log10(end), n_points)
    
    intervals = []
    
    if distribution_type == "histogram":
        # Adjacent pairs: bin intervals for histogram
        for i in range(len(points) - 1):
            intervals.append((points[i], points[i + 1]))
    
    elif distribution_type == "cumulative":
        # Start to each point: cumulative distribution
        for point in points:
            intervals.append((start, point))
        # Remove the first interval if it is (start, start) to avoid zero-length interval
        if intervals and intervals[0][0] >= intervals[0][1]:
            intervals.pop(0)
    
    elif distribution_type == "complementary_cumulative":
        # Each point to end: complementary cumulative distribution
        for point in points:
            intervals.append((point, end))
        # Remove the last interval if it is (end, end) to avoid zero-length interval
        if intervals and intervals[-1][0] >= intervals[-1][1]:
            intervals.pop()
    
    else:
        raise ValueError(
            f"Unknown distribution_type: '{distribution_type}'. "
            f"Must be 'histogram', 'cumulative', or 'complementary_cumulative'"
        )
    
    return intervals

# Define metric functions
def precision(tp, fp, fn):
    """Calculate precision: TP / (TP + FP)"""
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp, fp, fn):
    """Calculate recall/sensitivity: TP / (TP + FN)"""
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(tp, fp, fn):
    """Calculate F1 score: 2 * (precision * recall) / (precision + recall)"""
    return (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0