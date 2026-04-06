"""
BlendedCNV Pipeline - Main Entry Point

This script orchestrates the complete CNV analysis pipeline:
1. Computation: VCF conversion, consensus calling, benchmarking
2. Analysis: Statistical metrics, plots, and visualizations
"""

from utils import parse_args
from drivers.processing_driver import main as processing_main
from drivers.computation_driver import main as computation_main
from drivers.analysis_driver import main as analysis_main


def main():
    """
    Main pipeline entry point.
    Runs computation pipeline followed by analysis pipeline.
    """
    # Parse command-line arguments
    config, args = parse_args()

    do_processing = not (args.only_compute or args.only_analyze)
    do_computation = args.run_benchmark and not (args.only_process or args.only_analyze)
    do_analysis = args.run_benchmark and not (args.only_process or args.only_compute)
    
    print("\n" + "="*80)
    print("BLENDEDCNV PIPELINE - STARTING")
    print("="*80)
    
    # Step 1: Run processing pipeline
    if do_processing:
        print("\n" + "="*80)
        print("PHASE 1: PROCESSING PIPELINE")
        print("="*80)
        processing_main(config)

    # Step 2: Run computation pipeline
    if do_computation:
        print("\n" + "="*80)
        print("PHASE 2: COMPUTATION PIPELINE")
        print("="*80)
        computation_main(config)
    
    # Step 3: Run analysis pipeline
    if do_analysis:
        print("\n" + "="*80)
        print("PHASE 3: ANALYSIS PIPELINE")
        print("="*80)
        analysis_main(config)
    
    print("\n" + "="*80)
    print("BLENDEDCNV PIPELINE - COMPLETE")
    print("="*80)
    print(f"\nResults available in: {config['output_dir']}")


if __name__ == "__main__":
    main()
