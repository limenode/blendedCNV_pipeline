# consensuscnv

Graph-based consensus copy-number-variant calling, with an optional 
benchmarking mode that reproduces the evaluation in the accompanying paper.

Several CNV callers run on the same samples rarely agree. `consensuscnv` builds
the overlap graph across their outputs, collapses each connected component into
one call, and writes out every level of caller agreement — so with three callers
you get the calls one caller made, the calls two agreed on, and the calls all
three agreed on, as separate BED files.

<!-- [![Publication](https://img.shields.io/badge/DOI-[INSERT_DOI]-blue)]([INSERT_DOI_LINK]) -->

## What it does

- **Normalizes caller output.** Per-sample VCF/BCF from any number of CNV callers
  becomes one BED per sample per caller. Tool names are labels you choose.
- **Builds consensus.** Calls are joined when they clear a reciprocal-overlap
  threshold, and each connected component becomes one call spanning the union of
  its members. Every agreement level is written, and you can ask for several
  overlap thresholds in a single run.
- **Optional coordinate liftover** (e.g. hg18 to hg38) per dataset.
- **Optional benchmarking.** Scores every consensus set, every individual caller
  and any control call set against a merged truth set, and writes a metrics table
  plus, on request, the true/false-positive and false-negative calls themselves.

Output is in BED and CSV formats.

## Installation

Requires Python 3.12 or newer.

```bash
pip install consensuscnv
```

or

```bash
uv add consensuscnv
```

Two dependencies carry compiled extensions: `cyvcf2` (which needs htslib) and
`liftover`. If `pip` cannot build them, install those two from conda-forge first;
`pip` then finds them satisfied and installs the rest:

```bash
conda install -c conda-forge -c bioconda cyvcf2 liftover
pip install consensuscnv
```

To work on the code instead, clone the repository; it ships a `pixi.toml` that
pins the whole environment:

```bash
pixi install
pixi run pytest
```

## Quick start

Write a config template and the reference files it needs into a directory:

```bash
consensuscnv init template
```

This gives you `template/config.yaml`, the hg38 chromosome lengths
(`genome_primary_hg38.txt`, and `genome_autosome_hg38.txt` with chrX and chrY
commented out) and the hg38 excluded regions (`excluded_regions_hg38.bed`:
centromeres and assembly gaps). `config.yaml` holds just the fields `call`
needs, and `genome_file` / `excluded_regions_file` are already filled in with
the absolute paths of the written files, so the config works from any
directory. Point it at your data:

```yaml
experimental:
  "My Samples":
    cnvpytor: "/data/cnvpytor/{id}.calls.vcf"
    delly:    "/data/delly/{id}.bcf"
    gatk:     "/data/gatk/{id}/*_segments_{id}.vcf.gz"

output_dir: "/data/out"
```

`{id}` is the sample id, and glob patterns work. `genome_file` is a two-column
`name<TAB>length` file — a `.fa.fai` works — and it defines the analysis domain:
**a chromosome commented out with `#` is excluded from the run**, which is how
`genome_autosome_hg38.txt` drops the sex chromosomes. Any other build works the
same way; swap in its lengths and exclusion BED.

```bash
consensuscnv call template/config.yaml
```

This parses the VCFs and writes the consensus BEDs. Every option, with its
default and the settings `benchmark` reads, is documented inline in
`template/config.full.yaml`, written beside the short config.
The flags below can also be used to specify options, and take precedence over those inside of the config.

```
consensuscnv call config.yaml --reuse-beds          # skip parsing, re-merge only
consensuscnv call config.yaml --overlap 0.25,0.5    # one run per threshold
consensuscnv call config.yaml --min-size 500        # smallest call to keep
consensuscnv call config.yaml --per-sample          # one BED per sample
consensuscnv call config.yaml -o /somewhere/else
```

Parsing VCFs may take long depending on the input size, 
so the option `--reuse-beds` can be set to speed up repeated runs.

## Output

```
output_dir/
├── My_Samples/                          # parsed input, one directory per caller
│   ├── cnvpytor/{sample}.bed
│   ├── delly/{sample}.bed
│   └── gatk/{sample}.bed
├── consensus/                           # the output
│   └── My_Samples/
│       └── overlap_0.50/
│           ├── 1of3.bed                 # one caller or more
│           ├── 2of3.bed                 # two callers or more
│           └── 3of3.bed                 # all three
├── downloads/                           # cached truth sets fetched by URL
└── evaluation/                          # `consensuscnv benchmark` only
    ├── metrics.csv
    └── labels/                          # with --write-labels
```

Consensus BEDs contain the columns `chrom  start  end  svtype  source  sample_id`, 
where `source` is a pipe-joined list of the callers behind the call. Under `--per-sample` 
the sample column is dropped and the filename carries it instead.

The agreement levels **nest**: every call in `3of3.bed` also appears in
`2of3.bed`.

## Using the graph from Python

The consensus step is one use of a general primitive: a graph over intervals
whose edges carry reciprocal overlap (or gap distance), built once and filtered
at any threshold in a slice. `consensuscnv.callsets` builds it over one set of
calls, and `consensuscnv.classification` builds the equivalent between two sets.
Both are importable without a config.

Edges never cross chromosome, nor the fields named by `partition_by`. The
default, `("svtype", "sample_id")`, is what consensus calling needs. Passing
`("svtype",)` lets edges join samples, which is how a cohort is analysed:

```python
from consensuscnv.classification import IntervalSet, build_candidates, classify

# Load calls from BED file into IntervalSet object
intervals = IntervalSet.from_bed("out/consensus/Cohort/overlap_0.50/2of3.bed", genome="genome.txt")

# Which calls in one group of samples share a locus with a call in another?
affected = intervals.restrict_to_samples(affected_sample_names)
unaffected = intervals.restrict_to_samples(affected_sample_names, invert=True)

# Pairs may join samples but not SV types. Build once, classify at any threshold:
# an affected call with no partner is a "false positive" against the unaffected set.
pairs = build_candidates(affected, unaffected, partition_by=("svtype",))
result = classify(pairs, min_reciprocal_overlap=0.5, validate=False)
print(threshold, result.n_false_positive, "calls found only in affected samples")

# The calls themselves, by name, with a TP / FP label and partner count per row.
private = result.to_frame().query("label == 'FP'")

# Per-sample caller output works the same way, through a glob.
callers = IntervalSet.from_bed("out/Cohort/cnvpytor/*.bed")
```

`from_bed` takes one path, a glob, or a list of them; a six-column consensus
file carries its sample column, and a five-column per-sample file takes the
sample from its filename. Every `IntervalSet` reads back by name
(`chrom_names`, `svtype_names`, `sample_names`, `source_names`, `samples`) or as
a `DataFrame` (`to_frame()`); the integer `_idx` columns are what the package
computes with. Intervals from anywhere else go in through
`IntervalSet.from_records`, which takes `(chrom, start, end[, svtype[, source[,
sample_id]]])` tuples or mappings with those keys (a `DataFrame.to_dict("records")`
works) and fills in whatever a record leaves out:

```python
intervals = IntervalSet.from_records(rows, source="array", partition_by=("svtype",))
```

The full cross-sample graph, where each component is one locus in the cohort,
is one level down:

```python
from consensuscnv.callsets import collect_callsets, merge_components

loci = merge_components(collect_callsets("out/Cohort/cnvpytor/*.bed", partition_by=("svtype",)),
                        min_reciprocal_overlap=0.5)
```

Chromosome ids are assigned once per process, in the order of the first genome
loaded, so a later `genome=` must agree with it on the order of any shared name
(a subset or an appended superset is fine) or it raises. `reset_registries()`
in `consensuscnv.callsets` can be used to purge old assignments and orderings.

Gap edges are recorded only out to `search_radius`
(default 0, which still joins exactly-touching intervals), and filtering by a
wider `max_padding` raises rather than returning an incomplete answer. And a
merge whose edges cross samples has no per-locus sample, so reading one
(`merged.sample_idx`, `IntervalSet.from_merged`, the sample column of
`write_merged_bed`) raises; everything else on the merged set is defined.

## Reproducing the paper

```bash
consensuscnv benchmark config.yaml
```

This needs a `benchmark:` block naming one or more truth sets, as paths or URLs —
URLs are downloaded once into `output_dir/downloads/`, which is safe to delete. A
`control:` block adds a comparator such as SNP-array calls.

It parses everything, builds the consensus grid, and scores every (call set,
overlap, level) point plus each individual caller and each control against the
merged truth set, writing one row per call set to `evaluation/metrics.csv`. Add
`--write-labels` for the TP/FP/FN calls behind each row as BED files.

Every table and figure in the paper is produced by the scripts in `manuscript/scripts/`, 
each of which names in its docstring exactly which table or figure it produces. 
Run `parse_inputs.py` first if you want the parsing statistics behind Table 1.

### How calls are scored

- **True positive** — a call clearing `evaluation.reciprocal_overlap` (default
  0.5) against at least one truth interval.
- **False positive** — a call clearing it against none.
- **Precision** is defined as true positives over total calls.
- **Recall** is scored against the *whole* merged truth set. Truth sets are far
  larger than any call set, so recall carries a **ceiling** of
  `calls / truth intervals`, capped at one, and `metrics.csv` reports that
  ceiling next to it. A low recall against a low ceiling reflects the 
  limitations of a small call set; where matching is one-to-one, 
  recall divided by its ceiling is exactly precision.
- **F1** is derived from precision and recall.
- Matching may be **many-to-many**, so the number of calls that matched something
  (`n_true_positive`) and the number of truth intervals that were matched
  (`n_truth_found`) are different quantities. Both are reported, along with
  `one_to_one` for whether they coincide.
- The size floor (`consensus.min_size`) applies to the truth set as well as to
  the call sets, so both sides of a comparison cover the same size range.

## Development

```bash
pixi run pytest              # the test suite, synthetic and offline
pixi run ruff check src/ manuscript/ tests/
pixi run build               # sdist + wheel into dist/
```

Use `pixi run build` to produce a distribution.

## Included Files - Sources

The reference files `consensuscnv init` writes (kept in
`src/consensuscnv/templates/`) are derived from other databases. If you use
them, please cite the appropriate sources.

- `genome_primary_hg38.txt`
  - Human reference genome GRCh38/hg38 chromosome lengths for chr1-chr22, chrX, and chrY.
  - Extracted from file provided in the 1000 Genomes database, hosted by IGSR: `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai`
- `genome_autosome_hg38.txt`
  - The same file with chrX and chrY commented out, so a run using it is
    restricted to the autosomes. This is the file the paper's analysis uses.
- `excluded_regions_hg38.bed`
  - Output from performing `bedtools merge` between the following regions:
    - Centromeric regions from file provided in the 1000 Genomes database, hosted by IGSR: `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/20150713_location_of_centromeres_and_other_regions.txt`
    - Regions defined in the gap table provided by the UCSC hg38 database: `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/gap.txt.gz` 
- `included_regions_hg38.bed`
  - Genomic regions derived from a .bed representation of `genome_primary_hg38.txt` subtracted by `excluded_regions_hg38.bed`, performed using `bedtools subtract`.
  - Is not directly used in the pipeline. Provided to the user as a convenient reference to the regions of interest if using the other two files when setting up a config.

<!-- ## Citation

If you use this pipeline, please cite:

```
[Citation Pending]
```

## Contributing

[INSERT CONTRIBUTION GUIDELINES IF APPLICABLE] 

## Contact

- **Lionel Sequeira** - lionelsequeira@gmail.com

-->

## License

GPL-3.0-or-later. See [LICENSE](https://github.com/limenode/consensuscnv/blob/main/LICENSE).

<!-- ## Acknowledgments

- [INSERT FUNDING SOURCES]
- [INSERT COLLABORATORS/ACKNOWLEDGMENTS] -->
