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

<!-- Insert Workflow from Publication -->
```
Input VCFs → BED Conversion → Consensus Calling → Benchmark Processing → Binary Classification → Analysis
                                       ↓
                            Control Data Processing (Optional)
                                       ↓
                            Liftover (Optional)
```

### Step 1: BED Conversion
Converts VCF files from CNV calling tools to standardized BED format.

### Step 2: Consensus Calling
Generates intersection and union call sets across tools.

### Step 3: Control Processing
Parses SNP array data (e.g., PennCNV output) for comparison.

### Step 4: Liftover
Converts genomic coordinates between reference builds if needed.

### Step 5: Benchmark Evaluation
Classifies predictions as TP/FP/FN against gold-standard benchmarks.

### Step 6: Analysis
Generates performance metrics, plots, and visualizations.

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
# Input datasets (expects VCF-like format)
input:
  "Low Coverage": # Can assign any name desired with standard characters
    cnvpytor: "/path/to/low_cov/cnvpytor/{id}.calls.1000bp.vcf"
    delly: "/path/to/low_cov/delly/{id}.bcf"
    gatk: "/path/to/low_cov/gatk/{id}/*_segments_{id}.vcf.gz"
  "High Coverage":
    cnvpytor: "/path/to/high_cov/cnvpytor/{id}.calls.1000bp.vcf"
    delly: "/path/to/high_cov/delly/{id}.bcf"
    gatk: "/path/to/high_cov/gatk/{id}/*_segments_{id}.vcf.gz"

# Control datasets (optional, expects PennCNV-like output format)
control:
  "SNP Array": "/path/to/array/data.cnv"

# Benchmark datasets (expects VCF-like format)
benchmark_map:
  "1000G": "/path/to/hgsvc2/benchmark.vcf"
  "HGSVC3": "/path/to/hgsvc2/benchmark.bcf"
  "ONT Vienna": "/path/to/hgsvc2/benchmark.vcf.gz"

# Liftover specifications (optional)
liftover:
  "SNP Array":
    "from": "hg18"
    "to": "hg38"

# Output directory
output_dir: "/path/to/output"

# Reference genome file
genome_file: "/path/to/genome.txt"
```

### Running the Pipeline

#### Full Pipeline (Recommended)

Run the complete pipeline with a single command:

```bash
python src/main.py --config config.yaml
```

This will execute:
1. **Computation Pipeline**: VCF conversion, consensus calling, liftover, benchmarking
2. **Analysis Pipeline**: Statistical metrics, plots, and Venn diagrams

#### Individual Pipeline Components

You can also run each pipeline component separately:

```bash
# Run only computation pipeline
python src/computation_driver.py --config config.yaml

# Run only analysis pipeline
python src/analysis_driver.py --config config.yaml
```

## Output Structure

```
output_dir/
├── {input 1}/
│   ├── bed/                      # Per-tool BED files
│   ├── intersections/            # Intersection call sets
│   ├── unions/                   # Union call sets
│   └── binary_classification/    # TP/FP/FN classifications
├── {input 2...}/
│   └── [same structure as above]
├── {control 1}/
│   ├── bed/                      # Array-based CNV calls
│   └── binary_classification/
├── {control 2...}/
│   └── [same structure as above]
├── benchmark_parsing/            # Processed benchmark data
├── figures/                      # Contains plots for various analyses
└── venn_diagrams/                # Detection overlap diagrams
```

## Key Metrics

The pipeline computes:
- **Precision**: TP / (TP + FP)
- **Recall/Sensitivity**: TP / (TP + FN)
- **F1 Score**: 2 × TP / (2 × TP + FP + FN)

Metrics are generated across:
- CNV size distributions
- SV types (DEL, DUP)
- Consensus methods (intersections, unions)
- Sequencing depths

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

