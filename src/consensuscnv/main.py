"""BlendedCNV Pipeline - Entry Point"""

from consensuscnv.parsing.parsing_driver import parse_input_files
from consensuscnv.utils import parse_args


def main():
    config = parse_args()

    print("\n" + "="*80 + "\nBLENDEDCNV PIPELINE\n" + "="*80)

    print("\n" + "="*80 + "\nPHASE 1: PARSING PIPELINE\n" + "="*80)
    parse_input_files(config)

    # PHASE 2: COMPUTATION -- consensus calling + binary classification.
    # The old networkx implementation was archived at commit 70e52b4 and now
    # lives in archive/computation/. Its replacement is being built on the
    # CallSet layer in consensuscnv.callsets; wire it back in here once the
    # classification refactor lands. Until then the analysis phase reads
    # whatever classification output is already on disk.

    # print("\n" + "="*80 + "\nPHASE 3: ANALYSIS PIPELINE\n" + "="*80)
    # analysis_main(config)

    # print("\n" + "="*80 + "\nPIPELINE COMPLETED SUCCESSFULLY\n" + "="*80)
    # print(f"\nResults available in: {config.output_dir}")

if __name__ == "__main__":
    main()
