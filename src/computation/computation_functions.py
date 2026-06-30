import subprocess
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor

from output_layout import OutputLayout

def _run_subprocess_with_command(command: List[str]) -> None:
    subprocess.run(command, check=True)

def run_binary_classification_script(
    config: dict, 
    sets_for_classification: List[Tuple[str, str]]
):
    """
    Runs the binary classification script for each set of inputs defined in the configuration.
    Args:
        config: Configuration dictionary loaded from YAML
        sets_for_classification: List of tuples containing (input_path, output_path) for classification
    """
    layout = OutputLayout(Path(config['output_dir']))
    commands: List[List[str]] = []

    for input_path, output_path in sets_for_classification:
        if not Path(input_path).exists():
            print(f"Input path {input_path} does not exist. Skipping binary classification for this set.")
            continue

        commands.append([
            "./src/get_binary_classification.sh",
            input_path,
            output_path,
            str(layout.benchmark),
            config['genome_file'],
            str(config.get('matching_reciprocal_threshold', 0.5))
        ])

    with ProcessPoolExecutor() as executor:
        list(executor.map(_run_subprocess_with_command, commands))
