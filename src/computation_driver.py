from pathlib import Path
import json

from new_functions import parse_control_to_bed, parse_benchmarks_to_bed
from utils import parse_args
from computation_functions import run_binary_classification_script
from benchmark_handler import BenchmarkParser

def main(config: dict, debug: bool = False):
    """
    Main computation pipeline.
    
    Args:
        config: Configuration dictionary loaded from YAML
    """

    print("\nStep 3: Processing control datasets (SNP Array)...")
    # control_liftover_results = parse_control_to_bed(config)

    # if control_liftover_results:
    #     liftover_log = Path(config['output_dir']) / "logs" / "control_liftover_results.json"
    #     liftover_log.parent.mkdir(exist_ok=True, parents=True)
    #     with open(liftover_log, 'w') as f:
    #         json.dump(control_liftover_results, f, indent=4)



    print("\nStep 4: Parsing all benchmarks to BED format...")
    sample_data = parse_benchmarks_to_bed(config)
    print(sample_data["HG00096"])


    # if 'benchmark_map' not in config:
    #     print("No benchmark map found in config. Skipping computation pipeline.")
    #     return
    
    # # Initialize benchmark parser
    # benchmark_parser = BenchmarkParser(config['benchmark_map'])
    # output_dir = Path(config['output_dir'])
    # benchmark_subdir = output_dir / "benchmark_parsing"
    # benchmark_subdir.mkdir(exist_ok=True, parents=True)

    # benchmark_parser.parse_all_benchmarks_to_bed(
    #     benchmark_subdir, 
    #     common_samples_only=True, 
    #     genome_file_path=config['genome_file']
    # )

    # print("\nStep 5b: Performing liftover on benchmarks (if configured)...")
    # liftover_results = prepare_benchmark_bed_files_for_liftover(config)
    
    # if liftover_results:
    #     # Save liftover results to log file
    #     benchmark_liftover_log = output_dir / "logs" / "benchmark_liftover_results.json"
    #     benchmark_liftover_log.parent.mkdir(exist_ok=True, parents=True)
    #     with open(benchmark_liftover_log, 'w') as f:
    #         json.dump(liftover_results, f, indent=4)
    #     print("  Liftover completed and results saved.")
    # else:
    #     print("  No benchmarks require liftover.")

    # print("\nStep 5c: Merging parsed benchmarks across all benchmarks...")
    # merge_results = benchmark_parser.merge_across_benchmarks(
    #     benchmark_subdir, 
    #     genome_file_path=config['genome_file']
    # )
    
    # # Save merge results to log file
    # benchmark_log = output_dir / "logs" / "benchmark_processing_results.json"
    # benchmark_log.parent.mkdir(exist_ok=True, parents=True)
    # with open(benchmark_log, 'w') as f:
    #     json.dump(merge_results, f, indent=4)

    # print("\nStep 6: Running binary classification script...")
    # run_binary_classification_script(config)


if __name__ == "__main__":
    # Allow running standalone for testing
    config = parse_args()
    main(config)