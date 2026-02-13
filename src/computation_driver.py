import os
import sys
import subprocess
import yaml
import argparse

from pathlib import Path
from cnv_parser import CNVParser
from benchmark_handler import BenchmarkParser
from utils import parse_args

def _convert_vcfs_to_bed(config: dict):
    output_dir = Path(config['output_dir'])

    # For each input set, create a CNVParser instance and convert VCF files to BED format
    for key, input_map in config['input'].items():
        print(f"Converting input set: {key}")

        # Remove whitespace from key to create a valid directory name
        output_subdir_name = key.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name
        os.makedirs(output_subdir, exist_ok=True)

        # Create CNVParser instance and get all VCF files
        cnv_parser = CNVParser(input_map)
        all_vcf_files = cnv_parser.get_all_vcf_files()

        # Check that tool_patterns contains expected tools
        expected_tools = {"cnvpytor", "delly", "gatk"}
        if not expected_tools.issubset(set(cnv_parser.tool_patterns.keys())):
            print(f"Warning: Expected tools {expected_tools} not all found in tool patterns. Found: {set(cnv_parser.tool_patterns.keys())}")
            sys.exit(1)

        # Convert all VCF files and export to files
        for tool, id_path_pair in all_vcf_files.items():
            for sample_id, vcf_path in id_path_pair:
                data = cnv_parser.convert_vcf_to_bed(vcf_path)

                # Export to file
                output_prefix = output_subdir / "bed" / tool / sample_id
                os.makedirs(output_prefix.parent, exist_ok=True)
                output_prefix = str(output_prefix)

                data[data["svtype"] == "DEL"].to_csv(
                    output_prefix + ".DEL.bed", sep="\t", index=False, header=False
                )
                data[data["svtype"] == "DUP"].to_csv(
                    output_prefix + ".DUP.bed", sep="\t", index=False, header=False
            )   

def _run_consensus_calls_script(config: dict):
    output_dir = Path(config['output_dir'])

    for key, input_map in config['input'].items():
        print(f"Running consensus calls script for input set: {key}")
        # Remove whitespace from key to create a valid directory name
        output_subdir_name = key.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name

        command = [
            "./src/get_consensus_calls.sh",
            output_subdir / "bed/cnvpytor",
            output_subdir / "bed/delly",
            output_subdir / "bed/gatk",
            output_subdir,
        ]

        subprocess.run(command, check=True)

def _run_benchmark_processing_script(config: dict):
    
    benchmark_parser = BenchmarkParser(config['benchmark_map'])
    output_dir = Path(config['output_dir'])
    output_subdir = output_dir / "benchmark_parsing"
    os.makedirs(output_subdir, exist_ok=True)

    print("Parsing all benchmarks to BED format...")
    benchmark_parser.parse_all_benchmarks_to_bed(output_subdir, common_samples_only=True, genome_file_path=config['genome_file'])

    print("Merging parsed benchmarks across all benchmarks...")
    benchmark_parser.merge_across_benchmarks(output_subdir, genome_file_path=config['genome_file'])

def _run_binary_classification_script(config: dict):
    output_dir = Path(config['output_dir'])

    for key, input_map in config['input'].items():
        print(f"Running binary classification script for input set: {key}")
        # Remove whitespace from key to create a valid directory name
        output_subdir_name = key.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name

        command = [
            "./src/get_binary_classification.sh",
            output_subdir / "intersections",
            output_subdir / "binary_classification"/ "intersections",
            output_dir / "benchmark_parsing" / "merged",
            config['genome_file']
        ]

        subprocess.run(command, check=True)

        command = [
            "./src/get_binary_classification.sh",
            output_subdir / "unions",
            output_subdir / "binary_classification" / "unions",
            output_dir / "benchmark_parsing" / "merged",
            config['genome_file']
        ]

        subprocess.run(command, check=True)

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nStep 1: Converting VCF files to BED format...")
    _convert_vcfs_to_bed(config)

    print("\nStep 2: Running consensus calls script...")
    _run_consensus_calls_script(config)

    print("\nStep 3: Running benchmark processing script...")
    _run_benchmark_processing_script(config)

    print("\nStep 4: Running binary classification script...")
    _run_binary_classification_script(config)

    

if __name__ == "__main__":
    main()
