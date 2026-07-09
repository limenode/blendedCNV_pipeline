"""BlendedCNV Pipeline - Entry Point"""

from utils import parse_args
from parsing.parsing_driver import parse_input_files as parsing_main
from computation.computation_driver import main as computation_main
from analysis.analysis_driver import main as analysis_main

def main():
    config = parse_args()
    
    print("\n" + "="*80 + "\nBLENDEDCNV PIPELINE\n" + "="*80)

    print("\n" + "="*80 + "\nPHASE 1: PARSING PIPELINE\n" + "="*80)
    parsing_main(config)

    print("\n" + "="*80 + "\nPHASE 2: COMPUTATION PIPELINE\n" + "="*80)
    computation_main(config)

    print("\n" + "="*80 + "\nPHASE 3: ANALYSIS PIPELINE\n" + "="*80)
    analysis_main(config)
    
    print("\n" + "="*80 + "\nPIPELINE COMPLETED SUCCESSFULLY\n" + "="*80)
    print(f"\nResults available in: {config.output_dir}")

if __name__ == "__main__":
    main()
