# Changelog

## 0.2.0

The package became usable from a terminal. Before this release there was no
working entry point: `main.py` had been deleted while `[project.scripts]` still
pointed at it, `build_config()` raised on every config, and nothing in the
package ever called the consensus writer.

### Added

- **`consensuscnv call`** — parse the input call sets and write consensus BEDs.
  Every agreement level is written, from one caller up to however many are
  configured, and `consensus.reciprocal_overlap` accepts a list so a single run
  can produce several thresholds, each in its own directory. Flags:
  `--reuse-beds`, `--overlap`, `--min-size`, `--per-sample`, `-o/--output-dir`.
- **`consensuscnv benchmark`** — score every consensus set, every individual
  caller and every control against the merged truth set, writing
  `evaluation/metrics.csv`. `--write-labels` also writes the TP / FP / FN calls
  behind each row as BED files.
- `consensuscnv --version`.
- Config validation that reports **every** fault at once rather than failing on
  the first, and refuses call-set names that would collide with a directory the
  pipeline owns.
- A `consensus:` config block (`reciprocal_overlap`, `min_size`) and an
  `evaluation:` block (`benchmark_padding`, `reciprocal_overlap`).
- `max_excluded_fraction` as a config key.
- `tests/`, a synthetic offline suite covering the invariants the consensus
  driver rests on. `pixi run pytest`.

### Changed

- **Chromosome handling is driven entirely by `genome_file`.** `valid_chromosomes`
  and `chromosome_order` are gone, replaced by one ordered `chromosomes` list read
  from that file; commenting a chromosome out with `#` now excludes it from the
  run. The chromosome registry starts empty, so a run on a non-human genome
  carries nothing human in it.
- **`liftover:` keys resolve most-specific-first.** A key names the dataset whose
  files are in the wrong build: an experimental call set, a tool label inside one,
  a control, or a benchmark. Naming a tool lifts that caller wherever it appears;
  naming a call set lifts everything in it. Call-set keys previously did nothing.
- Benchmark sources given as URLs download when the benchmark path first needs
  them, into `output_dir/downloads/`, rather than during config loading — so a
  consensus run never touches the network. Downloads are staged through a `.part`
  file and renamed only on success.
- `scipy` moved to the runtime dependencies. It had been behind an optional extra
  while `callsets/merging.py` needed `scipy.sparse.csgraph`, so a plain install
  produced a package that could not merge a call set.
- Dev dependencies moved to a PEP 735 `[dependency-groups]` group.
- `[project.scripts]` points at `consensuscnv.cli:main`.

### Fixed

- **The PennCNV parser ignored its sample allowlist**, writing a BED for every
  sample in the input file rather than the requested ones.
- `build_config()` raised `TypeError` on every config.
- `benchmark_parser` dropped every record when the genome file was missing, where
  `penncnv_parser` kept them all; a missing genome file is now an error.
- The three parsers disagreed about which chromosomes were in scope: `vcf_parser`
  used a hardcoded `chr1`–`chr22` while the other two read the genome file.
- `max_excluded_fraction` had two different defaults for the same setting, 0.01
  through `parse_input_files` and 0.0 through the parsers themselves.
- A partially downloaded benchmark was left in place and reused on every later
  run.
- `sample_id_from_vcf` leaked its VCF handle and caught bare `Exception`.

### Removed

- `networkx`, and 16 unused packages from the development environment.
- The `benchmark` optional-dependency extra, empty once `scipy` moved to the
  runtime dependencies and plotting was ruled out of scope.
- Dead code: `SizeBinning`, `SizeMetrics`, `size_metrics`, `group_metrics`, and
  the metric helpers in `utils.py` — one of which, `recall`, was wrong, dividing
  a query-side count by a truth-side denominator.
- `src/test_*.py`. The cell scripts are superseded by `manuscript/scripts/`; the
  one invariant they asserted moved to `tests/`.

## 0.1.0

Initial parsing pipeline: VCF, PennCNV and benchmark sources normalized to BED.
