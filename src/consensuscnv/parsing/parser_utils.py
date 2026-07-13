import glob
from pathlib import Path

from consensuscnv.utils import PipelineConfig


def discover_samples_of_interest(config: PipelineConfig) -> set[str]:
    layout = config.layout
    
    sample_ids: set[str] = set()
    for key in config.experimental.keys():
        bed_paths = glob.glob(str(layout.set_dir(key)) + "/bed/*/*.bed")
        sample_ids |= {Path(path).name.split(".")[0] for path in bed_paths}

    if not sample_ids:
        print("Warning: No samples found in consensus call sets. Skipping control processing.")

    else:
        print(f"Found {len(sample_ids)} samples of interest from consensus call sets")

    return sample_ids