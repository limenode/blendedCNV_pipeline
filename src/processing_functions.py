import os
from pathlib import Path
import json
from cnv_parser import CNVParser
import subprocess

def convert_vcfs_to_bed(config: dict):
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

def run_consensus_calls_script(config: dict):
    output_dir = Path(config['output_dir'])
    results = {
        "consensus_1of3": {},
        "consensus_2of3": {},
        "consensus_3of3": {}
    }
    consensus_types = ["consensus_1of3", "consensus_2of3", "consensus_3of3"]

    for key, input_map in config['input'].items():
        print(f"Running consensus calls script for input set: {key}")
        # Remove whitespace from key to create a valid directory name
        output_subdir_name = key.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name

        tool_list = list(input_map.keys())

        if len(tool_list) != 3:
            print(f"  Warning: Expected exactly 3 tools for consensus calls, but found {len(tool_list)} for input set '{key}'. Skipping this input set.")
            continue

        # print(f"  Tools for this input set: {tool_list}")

        tools_and_names = [
            tool_list[0],
            str(output_subdir / "bed" / tool_list[0]),
            tool_list[1],
            str(output_subdir / "bed" / tool_list[1]),
            tool_list[2],
            str(output_subdir / "bed" / tool_list[2]),
        ]

        # Consensus calls 1/3 script
        command = [
            "./src/consensus_scripts/get_1of3_calls.sh",
            *tools_and_names,
            str(output_subdir / "consensus_1of3"),
            config['genome_file'],
            str(config.get('excluded_regions_file', "-"))
        ]
        subprocess.run(command, check=True)

        # Consnsus calls 2/3 script
        command = [
            "./src/consensus_scripts/get_2of3_calls.sh",
            *tools_and_names,
            str(output_subdir / "consensus_2of3"),
            config['genome_file'],
            str(config.get('excluded_regions_file', "-")),
            str(config.get('consensus_reciprocal_threshold', 0.5))
        ]
        subprocess.run(command, check=True)

        # Consensus calls 3/3 script
        command = [
            "./src/consensus_scripts/get_3of3_calls.sh",
            *tools_and_names,
            str(output_subdir / "consensus_3of3"),
            config['genome_file'],
            str(config.get('excluded_regions_file', "-")),
            str(config.get('consensus_reciprocal_threshold', 0.5))
        ]
        subprocess.run(command, check=True)

        # Read log files into results dictionary, and remove after reading
        for consensus_type in consensus_types:
            log_file = output_subdir / consensus_type / f"get_{consensus_type}_calls_summary.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    results[consensus_type][key] = json.load(f)
            else:
                print(f"Warning: Log file not found for consensus calls script: {log_file}")
            os.remove(log_file)
        
    # Save results to files
    for consensus_type in consensus_types:
        log_output_file = Path(config['output_dir']) / "logs" / f"{consensus_type}_results.json"
        os.makedirs(log_output_file.parent, exist_ok=True)
        with open(log_output_file, 'w') as f:
            json.dump(results[consensus_type], f, indent=4)
