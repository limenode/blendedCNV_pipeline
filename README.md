# Blended Genome-Exome (BGE) Sequencing CNV Detection Benchmark

A computational pipeline for benchmarking and analyzing Copy Number Variant (CNV) detection from low-pass, short-read Whole Genome Sequencing data derived from Blended Genome-Exome Sequencing.

<!-- [![Publication](https://img.shields.io/badge/DOI-[INSERT_DOI]-blue)]([INSERT_DOI_LINK]) -->

## Overview

This pipeline integrates CNV calls from multiple detection tools (CNVpytor, Delly, GATK-gCNV) across different sequencing depths and platforms, performs consensus calling, and benchmarks results against standard datasets.

## Features

- **Multi-tool integration**: Combines CNV detection results from CNVpytor, Delly, and GATK-gCNV in a 2/3 consensus call approach.
- **Control datasets**: Allows incorporation of SNP Microarray-based CNV calls from PennCNV to serve as a control.
- **Coordinate liftover**: Automatic genome build conversion (e.g., hg18 → hg38)
- **Benchmark evaluation**: Binary classification against reference datasets. Derives Precision, Recall/Sensitivity, and F1-score and generates density, cumulative, and complementary cumulative distribution plots.

## Pipeline Workflow

The pipeline consists of three distinct processing stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                      PROCESSING PIPELINE                        │
│  Input VCFs → BED Conversion → Consensus Calling                │
│                          ↓                                      │
│              Control Data Processing (Optional)                 │
│                          ↓                                      │
│                   Liftover (Optional)                           │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                     COMPUTATION PIPELINE                        │
│  Benchmark Download/Parsing → Binary Classification             │
│                    (TP/FP/FN Assignment)                        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                      ANALYSIS PIPELINE                          │
│  Performance Metrics → Distribution Plots → Venn Diagrams       │
└─────────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

**Purpose**: Prepare and standardize CNV call sets for evaluation

1. **BED Conversion**: Converts VCF files from CNV calling tools (i.e. CNVpytor, Delly, GATK-gCNV) to standardized BED format
2. **Consensus Calling**: Generates intersection and union call sets across tools using a 2/3 consensus approach
3. **Control Processing** *(Optional)*: Parses SNP array data (e.g., PennCNV output) for comparison
4. **Liftover** *(Optional)*: Converts genomic coordinates between reference builds (e.g., hg18 → hg38)

### Computation Pipeline

**Purpose**: Evaluate CNV calls against gold-standard benchmarks

1. **Benchmark Processing**: Downloads (if URL provided) and parses benchmark datasets
2. **Binary Classification**: Classifies predictions as True Positives (TP), False Positives (FP), or False Negatives (FN) against reference benchmarks

### Analysis Pipeline

**Purpose**: Generate performance metrics and visualizations

1. **Statistical Metrics**: Computes precision, recall/sensitivity, and F1-scores across CNV size distributions and SV types
2. **Distribution Plots**: Creates density, cumulative, and complementary cumulative distribution plots
3. **Venn Diagrams**: Visualizes detection overlap across tools and datasets

## Installation

### Requirements

- Python 3.8+
- Bedtools

### Setup

#### Unix/macOS

```bash
# Clone repository
git clone https://github.com/limenode/blendedCNV_pipeline.git
cd blendedCNV_pipeline

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

#### Windows

```bash
# Clone repository
git clone https://github.com/limenode/blendedCNV_pipeline.git
cd blendedCNV_pipeline

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

**Note:** Make sure Bedtools is installed and available in your system PATH. It should be accessable via the command `bedtools`.

## Usage

### Configuration

Create a YAML configuration file (see `config.yaml` example):

```yaml
# Input datasets
input:
  "Low Coverage":
    cnvpytor: "/path/to/low_cov/cnvpytor/{id}.calls.1000bp.vcf"
    delly: "/path/to/low_cov/delly/{id}.bcf"
    gatk: "/path/to/low_cov/gatk/{id}/*_segments_{id}.vcf.gz"
  "High Coverage":
    cnvpytor: "/path/to/high_cov/cnvpytor/{id}.calls.1000bp.vcf"
    delly: "/path/to/high_cov/delly/{id}.bcf"
    gatk: "/path/to/high_cov/gatk/{id}/*_segments_{id}.vcf.gz"

# Output directory
output_dir: "/path/to/output"

# Reference genome file
genome_file: "/path/to/genome.txt"

# Excluded regions file [optional]
excluded_regions_file: "/path/to/excluded_regions_hg38.bed"

# Control datasets [optional, used with --run-benchmark]
control:
  "SNP Array": "/path/to/array/data.cnv"

# Benchmark datasets [required when using --run-benchmark] (accepts URLs or paths)
benchmark_map:
  "1000G": "/path/to/hgsvc2/benchmark.vcf"
  "HGSVC3": "/path/to/hgsvc2/benchmark.bcf"
  "ONT Vienna": "/path/to/hgsvc2/benchmark.vcf.gz"

# Liftover specifications [optional]
liftover:
  "SNP Array":
    "from": "hg18"
    "to": "hg38"

# Reciprocal overlap threshold for generating consensus calls [optional] (default: 0.5)
consensus_reciprocal_threshold: 0.5

# Reciprocal overlap threshold for matching calls to benchmarks [optional] (default: 0.5)
matching_reciprocal_threshold: 0.5
```

### Running the Pipeline

#### Consensus Calling

Run the consensus calling on your input sets using the following command:

```bash
python src/main.py config.yaml
```

This will execute VCF conversion of evaluated call sets, consensus calling, and liftover if specified

#### Benchmarking

Run the complete benchmarking pipeline using the following command:

```bash
# Run the full benchmark pipeline
python src/main.py config.yaml --run-benchmark
```

This will execute the following:
1. **Computation Pipeline**: VCF conversion of benchmarks, binary classification of evaluated call sets
2. **Analysis Pipeline**: Generate statistical metrics tables, plots, and Venn diagrams

If you would like to only run one of the two parts of the benchmark pipeline, you can add the following arguments:
```bash
# Only run the computation sub-pipeline
python src/main.py config.yaml --run-benchmark --only-compute

# Only run the analysis sub-pipeline
python src/main.py config.yaml --run-benchmark --only-analyze
```

## Output Structure

```
output_dir/
├── {input 1}/
│   ├── bed/                      # Per-caller BED files for CNV calls
│   ├── conensus_1of3/            # Consensus calls, requires 1/3 caller agreement
│   ├── conensus_2of3/            # Consensus calls, requires 2/3 caller agreement
│   ├── conensus_3of3/            # Consensus calls, requires 1/3 caller agreement
│   └── binary_classification/    # Contains binary classification outputs per each call set (single callers + consensus calls)
│       └── {call set 1}/         # TP/FP/FN classifications
│       └── {call set 2...}/      
├── {input 2...}/
│   └── [same structure as above]
├── {control 1}/
│   ├── bed/                      # BED files for Array-based CNV calls 
│   └── binary_classification/    # TP/FP/FN classifications
├── {control 2...}/
│   └── [same structure as above]
├── benchmark_parsing/
│   └── merged/                   # BED files with CNV calls of all benchmarks merged together
├── figures/                      # Plots
└── logs/                         # Logs and tables
```

## Key Metrics for Benchmarking

- **True Positive**: A CNV in a call set with reciprocal overlap with at least one CNV in the benchmark call set that exceeds the threshold defined by `matching_reciprocal_threshold` in the configuration file (defaults to 0.50).
- **False Positive**: A CNV in a call set with insufficient reciprocal overlap with any of the CNVs in the benchmark call set.
- **False Negative**: A CNV in the benchmark call set that does not have any associated match in a call set. Any False Negative CNVs that are not detected by any of the call sets are deemed as "undiscoverable" and deducted from the total False Negative call set.
  - This behavior does not change absolute difference in FN between call sets but does change relative difference. This change allows for 1) the amplification of trend differences between call sets, 2) more comparable value ranges between precision and recall, and 3) prevents the F-score graph trends from being dominated by those of the recall graph due to extremely high FN values.

The pipeline computes:
- **Precision**: TP / (TP + FP)
- **Recall/Sensitivity**: TP / (TP + FN)
- **F_β Score**: ((1 + β^2) * TP) / (((1 + β^2) × TP) + (β^2 * FN) + FP)
  - Calculated for β = {1/2, 1, 2}

Metrics are generated across:
- CNV size distributions
- SV types (DEL, DUP)
- Consensus methods (intersections, unions)
- Sequencing depths

## Included Files - Sources

This repository hosts files in the `data/` directory that contains information derived from other databases. If you choose to use these files for this pipeline, please cite the appropriate sources.

- `genome_primary_hg38.txt`
  - Human reference genome GRCh38/hg38 chromosome lengths for chr1-chr22, chrX, and chrY.
  - Extracted from file provided in the 1000 Genomes database, hosted by IGSR: `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai`
- `excluded_regions_hg38.bed`
  - Output from performing `bedtools merge` between the following regions:
    - Centromeric regions from file provided in the 1000 Genomes database, hosted by IGSR: `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/20150713_location_of_centromeres_and_other_regions.txt`
    - Regions defined in the gap table provided by the UCSC hg38 database: `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/gap.txt.gz` 
- `included_regions.bed`
  - Genomic regions derived from a .bed representation of `genome_primary_hg38.txt` subtracted by `excluded_regions_hg38.bed`, performed using `bedtools subtract`.
  - Is not directly used in the pipeline. Provided to the user as a convenient reference to the regions of interest if using the other two files when setting up a config.

<!-- ## Citation

If you use this pipeline, please cite:

```
[Citation Pending]
```

## Contributing

[INSERT CONTRIBUTION GUIDELINES IF APPLICABLE]

## License

[INSERT LICENSE INFORMATION]

## Contact

- **Lionel Sequeira** - [lionelsequeira@gmail.com]

## Acknowledgments

- [INSERT FUNDING SOURCES]
- [INSERT COLLABORATORS/ACKNOWLEDGMENTS] -->

