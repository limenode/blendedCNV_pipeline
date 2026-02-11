import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Process CNV files from multiple tools')
    parser.add_argument('config', type=Path, help='Path to configuration YAML file')
    return parser.parse_args()