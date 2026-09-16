"""`consensuscnv init`: the template files reach the user, and the config they
get points at the files written beside it."""

import yaml

from consensuscnv.cli import TEMPLATE_FILES, main
from consensuscnv.genome import read_genome_file


def test_init_writes_every_template_and_fills_in_paths(tmp_path, capsys):
    target = tmp_path / "run"
    assert main(["init", str(target)]) == 0

    for name in TEMPLATE_FILES:
        assert (target / name).is_file(), name

    config = yaml.safe_load((target / "config.yaml").read_text())
    genome = target / "genome_autosome_hg38.txt"
    assert config["genome_file"] == str(genome.resolve())
    assert config["excluded_regions_file"] == str((target / "excluded_regions_hg38.bed").resolve())
    assert len(read_genome_file(genome)) == 22  # chrX and chrY are commented out

    out = capsys.readouterr().out
    assert out.count("wrote ") == len(TEMPLATE_FILES)


def test_init_refuses_to_overwrite_unless_forced(tmp_path, capsys):
    assert main(["init", str(tmp_path)]) == 0
    (tmp_path / "config.yaml").write_text("mine: true\n")

    assert main(["init", str(tmp_path)]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert (tmp_path / "config.yaml").read_text() == "mine: true\n"

    assert main(["init", str(tmp_path), "--force"]) == 0
    assert "genome_file" in (tmp_path / "config.yaml").read_text()
