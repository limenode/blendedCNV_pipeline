# Changelog

## 0.3.0

The first release on PyPI: `pip install consensuscnv`. The overlap graph became
a resource in its own right rather than an internal of consensus calling. Both
graph builders now take the same two build-time choices, record them on the
result, and refuse downstream what the choices no longer guarantee.

### Added

- **`consensuscnv init [dir]`** writes a config template and the reference
  files it points at -- the hg38 chromosome lengths, with and without the sex
  chromosomes, and the hg38 excluded regions -- with `genome_file` and
  `excluded_regions_file` already filled in. `--force` overwrites. These
  files ship inside the package (`consensuscnv/templates/`), so a `pip`
  install has everything the quick start needs.
- `python -m consensuscnv`, the same entry point as the script.
- A GitHub Actions workflow that builds one distribution, verifies it on
  Python 3.12--3.14 against the test suite and both entry points, and
  publishes to TestPyPI on a manual run and to PyPI on a GitHub release
  through trusted publishing.
- **`partition_by`** on `build_callset`, `collect_callsets`, `build_candidates`
  and `partition_ids`: the `Call` fields an edge never crosses, from
  `PARTITION_FIELDS = ("svtype", "sample_id")`; chromosome always partitions.
  The default is unchanged and is what consensus calling needs. Dropping
  `"sample_id"` gives a cross-sample graph over a cohort -- which calls in one
  group of samples share a locus with calls in another, at any threshold -- and
  the two builders give identical answers to that question.
- **`search_radius`** on `build_callset` and `collect_callsets`, mirroring
  `build_candidates`: the widest gap a recorded edge spans, default 0. Gap
  edges are now complete out to the radius, and `filter_edges` refuses a
  `max_padding` beyond it rather than returning a silently incomplete selection.
- `CallSet.require_partition`, and `MergedCallSet.chrom_idx` / `svtype_idx`
  alongside `sample_idx`. The per-component reads raise when the parent does
  not partition on that field; `IntervalSet.from_merged` and
  `write_merged_bed(include_sample=True)` go through them.
- **`IntervalSet.from_bed`** and **`IntervalSet.from_records`**: one call from
  files -- a path, a glob, or a list of them -- or plain records to an
  IntervalSet whose `origin` carries the built graph. `genome=` seeds the
  chromosome registry, so a script needs one import. Underneath,
  `seed_chromosomes` takes a genome file path and `collect_callsets` takes
  paths and globs alongside CallSets and Calls, singly or in a list.
- **Names at the surface.** `IntervalSet.chrom_names` / `svtype_names` /
  `sample_names` / `source_names` / `samples`, and `to_frame()` for a
  `DataFrame` in the BED layout; `Classification.to_frame(side)` adds the
  ``TP`` / ``FP`` label (or ``found``) and partner count per row.
  `restrict_to_samples` takes names as well as ids, and `invert=True` keeps
  the other samples. The registries are an implementation detail again.
- **`reset_registries`**, and `seed_chromosomes` raises when a later genome
  orders shared names differently from the ids already assigned -- rows would
  sort by the new order but be written in the old one. A subset, or a superset
  that only appends, is accepted as before.
- **`calls_from_records`**: `Call`s from `(chrom, start, end[, svtype[, source[,
  sample_id]]])` tuples or mappings, with defaults for whatever a record leaves
  out, so intervals from anywhere go into `build_callset` without a BED.
- `read_bed_calls(sample_id=...)` to name the sample explicitly.
- `read_genome_file` is exported from `consensuscnv.callsets` next to
  `seed_chromosomes` (it lives in the new `consensuscnv.genome`; the `utils`
  name still works), and `consensuscnv.classification` exports its public
  names, so each layer is one import.
- `tests/test_partition.py`, and BED I/O tests for the sample column and
  `calls_from_records`.
- A "Using the graph from Python" section in the README.

### Changed

- **Python 3.12 or newer is required.** The package declared 3.10 but its
  pinned numpy and scipy need 3.12; the metadata now says so.
- The config template and the hg38 reference files moved from the repository
  root (`config.yaml`, `data/`) into the package, where `init` reads them.
- The sdist carries the runnable test suite (`tests/conftest.py` was missing,
  so the tests it shipped could not run) and the changelog.
- The package version is defined once, in `pyproject.toml`; `pixi.toml` no
  longer carries a copy.
- `build_callset` keeps, per partition, only the calls a later call could still
  reach -- those ending within `search_radius` of the current start -- instead of
  the chain of overlapping-or-touching calls. Consensus output is identical, and
  the default build is 15--20% faster because it no longer records gap edges it
  could not serve. Before this the gap list was complete only at distance 0:
  a pair further apart was recorded only if an unbroken chain of touching calls
  ran between them. Connected components were nevertheless exact at every
  padding, because every missing pair is bridged by recorded shorter ones
  (verified against a complete graph on the 327,785-record truth set at 1 kb and
  10 kb), so no reported number changes. The edge list itself was not a
  complete pair list, which matters once it is exposed.
- `evaluation.py` and the padding-sweep scripts (`benchmark_padding.py`,
  `sensitivity.py`) build the truth set with the radius they go on to filter at.

### Fixed

- **`read_bed_calls` could not read the pipeline's own combined output.** It
  took the sample from the filename, so `2of3.bed` read back with every call
  assigned to a sample called `2of3`. A sixth column is now the sample.
- `Registry.get` documented `None` for a missing name; it returns `-1`.
- The extra-samples warning from `classify(validate=True)` reported the wrong
  truth-sample count and listed registry ids; it now counts correctly and
  names the samples.

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
