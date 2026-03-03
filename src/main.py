#!/usr/bin/env python3
"""
BlendedCNV Pipeline - Main Entry Point

This script orchestrates the complete CNV analysis pipeline:
1. Computation: VCF conversion, consensus calling, benchmarking
2. Analysis: Statistical metrics, plots, and visualizations
"""

import yaml
from utils import parse_args
from processing_driver import main as processing_main
from computation_driver import main as computation_main
from analysis_driver import main as analysis_main


def main():
    """
    Main pipeline entry point.
    Runs computation pipeline followed by analysis pipeline.
    """
    # Parse command-line arguments
    args = parse_args()
    
    debug = True

    # Load configuration from YAML file
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*80)
    print("BLENDEDCNV PIPELINE - STARTING")
    print("="*80)
    


    # Step 1: Run processing pipeline
    print("\n" + "="*80)
    print("PHASE 1: PROCESSING PIPELINE")
    print("="*80)
    processing_main(config, debug=debug)

    # Step 2: Run computation pipeline
    print("\n" + "="*80)
    print("PHASE 2: COMPUTATION PIPELINE")
    print("="*80)
    computation_main(config, debug=debug)
    
    # Step 3: Run analysis pipeline
    print("\n" + "="*80)
    print("PHASE 3: ANALYSIS PIPELINE")
    print("="*80)
    analysis_main(config, debug=debug)
    
    print("\n" + "="*80)
    print("BLENDEDCNV PIPELINE - COMPLETE")
    print("="*80)
    print(f"\nResults available in: {config['output_dir']}")


if __name__ == "__main__":
    main()
