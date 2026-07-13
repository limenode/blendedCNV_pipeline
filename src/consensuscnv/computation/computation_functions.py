import subprocess
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor

from consensuscnv.utils import PipelineConfig

_CLASSIFICATION_SCRIPT = Path(__file__).parent / "get_binary_classification.sh"

def _run_subprocess_with_command(command: List[str]) -> None:
    subprocess.run(command, check=True)

def run_binary_classification_script(
    config: PipelineConfig,
    sets_for_classification: List[Tuple[str, str]]
):
    """
    Runs the binary classification script for each set of inputs defined in the configuration.
    Args:
        config: Configuration dictionary loaded from YAML
        sets_for_classification: List of tuples containing (input_path, output_path) for classification
    """
    layout = config.layout
    commands: List[List[str]] = []

    for input_path, output_path in sets_for_classification:
        if not Path(input_path).exists():
            print(f"Input path {input_path} does not exist. Skipping binary classification for this set.")
            continue

        commands.append([
            str(_CLASSIFICATION_SCRIPT),
            input_path,
            output_path,
            str(layout.benchmark_dir("merged")),
            config.genome_file,
            str(config.matching_reciprocal_threshold)
        ])

    with ProcessPoolExecutor() as executor:
        list(executor.map(_run_subprocess_with_command, commands))
