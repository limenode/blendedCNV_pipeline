"""
BlendedCNV Pipeline - Entry Point

This script orchestrates the complete CNV analysis pipeline:
1. Computation: VCF conversion, consensus calling, benchmarking
2. Analysis: Statistical metrics, plots, and visualizations
"""

from utils import parse_args
from processing.processing_driver import main as processing_main
from computation.computation_driver import main as computation_main
from analysis.analysis_driver import main as analysis_main


def main():
    """
    Pipeline entry point.
    """
    # Parse command-line arguments
    config = parse_args()

    print("\n" + "="*80 + "\nBLENDEDCNV PIPELINE\n" + "="*80)

    # Step 1: Run processing pipeline
    if config.do_processing:
        print("\n" + "="*80 + "\nPHASE 1: PROCESSING PIPELINE\n" + "="*80)
        processing_main(config)

    # Step 2: Run computation pipeline
    if config.do_computation:
        print("\n" + "="*80 + "\nPHASE 2: COMPUTATION PIPELINE\n" + "="*80)
        computation_main(config)

    # Step 3: Run analysis pipeline
    if config.do_analysis:
        print("\n" + "="*80 + "\nPHASE 3: ANALYSIS PIPELINE\n" + "="*80)
        analysis_main(config)
    
    print("\n" + "="*80 + "\nPIPELINE COMPLETED SUCCESSFULLY\n" + "="*80)
    print(f"\nResults available in: {config.output_dir}")


if __name__ == "__main__":
    main()
