import os
import json
import subprocess

from utils import PipelineConfig

def run_consensus_calls_script(config: PipelineConfig):
    layout = config.layout
    results = {
        "consensus_1of3": {},
        "consensus_2of3": {},
        "consensus_3of3": {}
    }
    consensus_types = ["consensus_1of3", "consensus_2of3", "consensus_3of3"]

    for key, input_map in config.input.items():
        print(f"Running consensus calls script for input set: {key}")

        tool_list = list(input_map.keys())

        if len(tool_list) != 3:
            print(f"  Warning: Expected exactly 3 tools for consensus calls, but found {len(tool_list)} for input set '{key}'. Skipping this input set.")
            continue

        # print(f"  Tools for this input set: {tool_list}")

        tools_and_names = [
            tool_list[0],
            str(layout.bed_tool_dir(key, tool_list[0])),
            tool_list[1],
            str(layout.bed_tool_dir(key, tool_list[1])),
            tool_list[2],
            str(layout.bed_tool_dir(key, tool_list[2])),
        ]

        # Consensus calls 1/3 script
        command = [
            "./src/consensus_scripts/get_1of3_calls.sh",
            *tools_and_names,
            str(layout.consensus_dir(key, 1)),
            config.genome_file
        ]
        subprocess.run(command, check=True)

        # Consnsus calls 2/3 script
        command = [
            "./src/consensus_scripts/get_2of3_calls.sh",
            *tools_and_names,
            str(layout.consensus_dir(key, 2)),
            config.genome_file,
            str(config.consensus_reciprocal_threshold)
        ]
        subprocess.run(command, check=True)

        # Consensus calls 3/3 script
        command = [
            "./src/consensus_scripts/get_3of3_calls.sh",
            *tools_and_names,
            str(layout.consensus_dir(key, 3)),
            config.genome_file,
            str(config.consensus_reciprocal_threshold)
        ]
        subprocess.run(command, check=True)

        # Read log files into results dictionary, and remove after reading
        for consensus_type in consensus_types:
            log_file = layout.set_dir(key) / consensus_type / f"get_{consensus_type}_calls_summary.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    results[consensus_type][key] = json.load(f)
            else:
                print(f"Warning: Log file not found for consensus calls script: {log_file}")
            os.remove(log_file)

    # Save results to files
    for consensus_type in consensus_types:
        log_output_file = layout.log(f"{consensus_type}_results.json")
        os.makedirs(log_output_file.parent, exist_ok=True)
        with open(log_output_file, 'w') as f:
            json.dump(results[consensus_type], f, indent=4)
