import subprocess
from pathlib import Path

def run_binary_classification_script(config: dict):
    output_dir = Path(config['output_dir'])

    if not (output_dir / "benchmark_parsing" / "merged").exists():
        print("Merged benchmark file not found. Skipping binary classification step.")
        return

    for key, input_map in config['input'].items():
        print(f"Running binary classification script for input set: {key}")
        # Remove whitespace from key to create a valid directory name
        output_subdir_name = key.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name

        command = [
            "./src/get_binary_classification.sh",
            str(output_subdir / "consensus_2of3" / "intersections"),
            str(output_subdir / "binary_classification" / "intersections"),
            str(output_dir / "benchmark_parsing" / "merged"),
            config['genome_file'],
            str(config.get('matching_reciprocal_threshold', 0.5))
        ]

        subprocess.run(command, check=True)

        command = [
            "./src/get_binary_classification.sh",
            str(output_subdir / "consensus_2of3" / "unions"),
            str(output_subdir / "binary_classification" / "unions"),
            str(output_dir / "benchmark_parsing" / "merged"),
            config['genome_file'],
            str(config.get('matching_reciprocal_threshold', 0.5))
        ]

        subprocess.run(command, check=True)
    
    for key, input_map in config['control'].items():
        print(f"Running binary classification script for control dataset: {key}")
        # Remove whitespace from key to create a valid directory name
        output_subdir_name = key.replace(" ", "_")
        output_subdir = output_dir / output_subdir_name

        command = [
            "./src/get_binary_classification.sh",
            str(output_subdir / "bed"),
            str(output_subdir / "binary_classification"),
            str(output_dir / "benchmark_parsing" / "merged"),
            config['genome_file'],
            str(config.get('matching_reciprocal_threshold', 0.5))
        ]

        subprocess.run(command, check=True)
