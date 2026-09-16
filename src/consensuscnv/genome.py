"""Reading the genome file: the ordered chromosome names over which an analysis is performed."""

from pathlib import Path


def read_genome_file(path: str | Path) -> tuple[str, ...]:
    """Ordered, de-duplicated chromosome names from a genome/faidx-style file.

    Reads column 0 of each line. Blank lines and lines whose first non-whitespace
    character is ``#`` are skipped, so commenting a chromosome out of the genome
    file is the supported way to drop it from the analysis.
    """
    names: dict[str, None] = {}  # insertion-ordered, and de-duplicates
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names[line.split()[0]] = None

    if not names:
        raise ValueError(f"No chromosomes found in genome file: {path}")
    return tuple(names)
