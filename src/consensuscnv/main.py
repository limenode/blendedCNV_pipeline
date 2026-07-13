"""BlendedCNV Pipeline - Entry Point"""

from consensuscnv.analysis.analysis_driver import main as analysis_main
from consensuscnv.computation.computation_driver import main as computation_main
from consensuscnv.parsing.parsing_driver import parse_input_files
from consensuscnv.utils import parse_args


def main():
    config = parse_args()
    
    print("\n" + "="*80 + "\nBLENDEDCNV PIPELINE\n" + "="*80)

    print("\n" + "="*80 + "\nPHASE 1: PARSING PIPELINE\n" + "="*80)
    parse_input_files(config)

    print("\n" + "="*80 + "\nPHASE 2: COMPUTATION PIPELINE\n" + "="*80)
    computation_main(config)

    print("\n" + "="*80 + "\nPHASE 3: ANALYSIS PIPELINE\n" + "="*80)
    analysis_main(config)
    
    print("\n" + "="*80 + "\nPIPELINE COMPLETED SUCCESSFULLY\n" + "="*80)
    print(f"\nResults available in: {config.output_dir}")

if __name__ == "__main__":
    main()
