// =============================================================================
// Evaluating Low-Pass Whole Genome Sequencing as a Cost-Effective Method
// for Copy Number Variant Detection
//
// Prose uses semantic line breaks: one sentence per line, no column limit.
// =============================================================================

#set document(
  title: "Evaluating Low-Pass Whole Genome Sequencing as a Cost-Effective Method for Copy Number Variant Detection",
  author: ("Lionel Sequeira", "Thomas V Fernandez", "Gary A Heiman", "Jinchuan Xing"),
)

#set page(paper: "us-letter", margin: 1in, numbering: "1")
#set text(font: "Libertinus Serif", size: 11pt, lang: "en")
#set par(justify: true, leading: 0.72em, spacing: 1.1em)
#set heading(numbering: none)

#show heading.where(level: 1): it => block(above: 1.8em, below: 0.9em)[
  #set text(size: 15pt, weight: "bold")
  #it.body
]
#show heading.where(level: 2): it => block(above: 1.4em, below: 0.7em)[
  #set text(size: 12.5pt, weight: "bold")
  #it.body
]
#show heading.where(level: 3): it => block(above: 1.2em, below: 0.6em)[
  #set text(size: 11pt, weight: "bold", style: "italic")
  #it.body
]
#show link: set text(fill: rgb("#1a4f8a"))
#show table.cell.where(y: 0): strong
#set table(stroke: (x, y) => (
  top: if y <= 1 { 0.6pt } else { 0pt },
  bottom: 0.6pt,
))

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

// Citation key. Placeholder for a real bibliography: swapping this one
// definition for `#cite(label(k))` converts the whole manuscript at once.
#let c(body) = [#box[\[#body\]]]

// Figure placeholder. Replace each call with e.g.
//   #image("figures/figure1.png", width: 100%)
#let figplaceholder(name, height: 5cm) = rect(
  width: 100%,
  height: height,
  stroke: (paint: luma(160), dash: "dashed", thickness: 0.6pt),
  fill: luma(248),
  align(center + horizon)[
    #text(fill: luma(110), style: "italic", size: 10pt)[image placeholder — #name]
  ],
)

// Caption styled like the source document (bold label, italic body),
// with numbering preserved verbatim from the draft rather than auto-generated.
#let cap(label, body) = block(width: 100%, above: 0.7em)[
  #set text(size: 9.5pt)
  #strong[#emph[#label]] #emph[#body]
]

// -----------------------------------------------------------------------------
// Title block
// -----------------------------------------------------------------------------

#align(center)[
  #block(width: 100%)[
    #text(size: 17pt, weight: "bold")[
      Evaluating Low-Pass Whole Genome Sequencing as a Cost-Effective
      Method for Copy Number Variant Detection
    ]
  ]

  #v(0.8em)

  Lionel Sequeira#super[1,2],
  Thomas V Fernandez#super[3,4],
  Gary A Heiman#super[1,2],
  Jinchuan Xing#super[1,2]
]

#v(0.6em)

#set text(size: 10pt)
#super[1] Department of Genetics, Rutgers, The State University of New Jersey, Piscataway, NJ 08854, USA

#super[2] Human Genetics Institute of New Jersey, Rutgers, The State University of New Jersey, Piscataway, NJ 08854, USA

#super[3] Child Study Center, Yale School of Medicine, New Haven, CT 06510, USA

#super[4] Department of Psychiatry, Yale School of Medicine, New Haven, CT 06510, USA
#set text(size: 11pt)

#v(0.8em)

*Correspondence:*

#block(
  width: 100%,
  stroke: 0.6pt + luma(140),
  inset: 10pt,
)[
  Jinchuan Xing, Ph.D. \
  Department of Genetics \
  Rutgers, The State University of New Jersey \
  Life Science Building 225 \
  145 Bevier Road \
  Piscataway, NJ 08854 \
  Email: #link("mailto:jinchuan.xing@rutgers.edu")[jinchuan.xing\@rutgers.edu]
]

= Recent additions to the manuscript:
- Note: You can click on the blue text to move directly to the location of interest in the manuscript.
- Changed #link(<m_benchmark_prep>)[Benchmark Dataset Preparation].
  - Wrote down explicit logic for which variant types from the VCF files were extracted into the BED files and while variant types were left behind.
  - Have not chosen to add any new sets yet to fill in the lack of DUP records that we now have until further discussion.
- Changed #link(<m_cnv_overlap>)[CNV Overlap and Adjacency Graph Building] and #link(<m_consensus_calling>)[Consensus CNV Calling].
  - Reflect the graph-based approach that we now take to parse and merge the calls.
- Added #link(<m_null_model>)[Null Model for Caller Agreement].
  - This proposes that if all of the calls in the aggregate call set were all part of a single population with a predefined probability of being detected by a caller, then you should be able to model the single-caller counts from the two-caller and three-caller consensus counts.
  - Rejecting this allows us to propose that there exists at least two populations of calls with different behaviors, with one being primarily composed of calls that are detectable independent of caller algorithm, and the other being primarily composed of calls that were identified due to caller specific effects, whether that be incorrect calls, artifact capture, algorithmic differences, etc.
- Expanded call set parsing and analysis sections of Results, see #link(<r_input_call_sets>)[Input Call Sets After Parsing] and #link(<r_sequence_consensus>)[Sequence-based Consensus Call Set Construction].
  - Before, these sections were combined into one and much shorter because of how trivial our old operations were. Now we have considerably more information so I decided to split the sections up between statistics on the raw counts from all call set sources after the initial parsing step, and the consensus calling which now includes the new null model for identifying the populations of calls within the consensus set.
  - The Venn diagram for the Consensus section has been updated, and now there are tables for the breakdown of all 7 parts of the Venn diagram (3 for single-callers, 3 for pairwise combinations, and 1 for all callers).
- Added parameterization section, see #link(<r_parameterizing>)[Parameterizing the Comparison].
  - This includes four subsections: Benchmark Padding, Size Floor, Consensus reciprocal overlap threshold, and Classification reciprocal overlap threshold.
  - The performance of the query/benchmark comparison is analyzed across the appropriate range for each parameter of interest.
  - This is coupled with a subsequent section on the Variance-based sensitivity analysis to show that the combined effects of parameter combinations are not significant enough to warrant a combinatorial analysis, and that analyzing each parameter one at a time is sufficient enough.
- Added sensitivity analysis and Pareto front sections in Methods, see #link(<m_vbsa_pf>)[Variance-based sensitivity analysis and Pareto front].
  - This includes the equations for deriving the first-order Sobol index, and how this corresponds to the contribution that each parameter has to a given performance metric.
  - The aggregation of the first order Sobol indices is a measure of the extent to which the one-at-a-time profiles describe the behavior of the pipeline compared to the joint effects of multiple parameters. If the first order Sobol indices add to close to 1, that means that the combined affects of parameters are negligible and that the one-at-a-time profiles are reasonable methods to determine individual parameter optimums.
  - The Pareto front shows all of the parameter combinations that yield the best performance assuming that a subset of other parameters are leaved fixed. This discards all combinations that are strictly worse than others, and yields a "front" of combinations. We show the Pareto front on a Precision vs. Recall graph, which points corresponding to one combination of four different parameters (benchmark padding, size floor, consensus reciprocal overlap, classification reciprocal overlap).

= Abstract

// [To be written.]

= Introduction

Copy number variation (CNV) is a type of structural variation (SV) characterized by duplications or deletions of genomic regions, with the number of repeats varying among individuals.
CNVs are typically distinguished from SVs as variations that are greater than 1kb in length.
CNVs play major roles in genetic diversity and disease phenotypes, and account for approximately 4.8-9.7% of the human genome #c[Zarrei 2015].
CNVs have been implicated as the primary contributors to the etiology of major diseases #c[Weischenfeldt 2013] #c[Hu 2018] #c[Glessner 2020], such as psychiatric disorders #c[Kushima 2025], cancers #c[Beroukhim 2010] #c[Shlien 2009], and rare genetic disorders #c[Lemire 2024].
Large CNVs have been tested clinically, although the clinical interpretation can be challenging #c[Hu 2018] #c[Nowakowska 2017].
In addition, our understanding of CNVs' functional impact and means of reliable detection are still limited #c[Valsesia 2013].

CNVs are typically detected using two approaches: array-based methods that use microarray Comparative Genomic Hybridization (aCGH) or SNP genotyping microarray to infer CNVs, and sequence-based methods that use next generation sequencing (NGS) or long read sequencing techniques to derive copy number from whole-exome sequencing (WES) and/or whole-genome sequencing (WGS) data.
Array-based methods are generally cheaper and easier to perform, but suffer from fluorescence noise, non-specific hybridization, and poor breakpoint prediction #c[Valsesia 2013] #c[Li and Olivier 2012].
Conversely, sequencing-based methods can achieve single-base breakpoint resolution and detect inversions, translocations, and _de novo_ CNVs that array-based methods struggle to identify.
However, sequencing-based workflows that rely on high-coverage genomes are more expensive.
Additionally, detection algorithms can fail to detect CNV regions due to non-uniform coverage.
It may also yield high false positive rates, especially when using short-read data, due to erroneous mapping on highly repetitive genomic regions #c[Li and Olivier 2012].
Additionally, analyzing high-coverage sequencing data necessitates significant hardware infrastructure to store and process the data.

Recently, the Blended Genome-Exome (BGE) sequencing approach has emerged as a cost-competitive method that allows for obtaining low-coverage WGS data in addition to the WES data from the same sample.
This process involves sequencing both genome and exome libraries of a sample using an NGS platform to generate low-coverage whole genome sequencing (lcWGS, 1-4x mean depth) data, in addition to the high-coverage WES data (30-40x) #c[Boltz 2026].
This protocol allows a minimum incurrence of additional cost as it can be integrated into existing WES pipelines #c[Broad Clinical Labs] #c[DeFelice 2024].
BGE has been shown to be valuable in performing cost-effective imputation and rare variant inference in an easily scalable manner.
This is particularly appealing for large population studies where deep whole genomes are not economically feasible and for particular cohorts for which arrays fail to capture critical population-specific genetic variation #c[DeFelice 2024].
Additionally, forgoing the usage of SNP microarray data avoids additional costs and complexities with harmonizing array data and sequencing data #c[DeFelice 2024].
BGE data has also shown to be a promising method to capture unbiased genetic diversity in underrepresented populations at a fraction of the cost of deep WGS, with performance that is competitive against population-specific GWAS arrays #c[Boltz 2026].

In addition to the small variant imputation, the lcWGS has also been shown to be a cost-effective way for CNV calling in several studies #c[Kucharík 2021] #c[Mazzonetto 2024] #c[Mazzonetto 2024 (2)].
However, a comprehensive evaluation of the performance of lcWGS-based CNV detection derived from a BGE sequencing approach is still lacking.
While BGE-derived exome data supports accurate detection of protein-coding CNVs #c[Boltz 2026], the performance of genome-wide CNV calling from the accompanying low-pass genome reads has not been systematically evaluated.
In this study, we sought to evaluate the effectiveness of using short-read lcWGS sequencing data, such as that from BGE, for detection of CNVs traditionally targeted by SNP microarrays and develop a best-practice workflow for CNV calling using lcWGS data.

= Materials & Methods

== Sequence Data and Reference Files Retrieval

Thirteen samples from the 1000 Genomes project were selected for the analysis because of their presence in the three benchmark SV sets (1000 Genomes SV benchmark set #c[1000G 2015], HGSVC3 #c[Logsdon 2025], and Oxford Nanopore Technology (ONT) Vienna set #c[Schloissnig 2024]), and because the availability of their SNP genotyping array data.
The 13 samples can be identified on the International Genome Sample Resource (IGSR) data portal website (#link("https://www.internationalgenome.org/data-portal/sample")), using the following filters: 1) data collections: "1000 Genomes 30x on GRCh38", "Human Genome Structural Variation Consortium, Phase 3", "1000 Genomes phase 3 release", and "1KG_ONT_VIENNA", and 2) technology "HD genotype chip".

High-coverage (\~30x) short-read WGS data aligned to the GRCh38 reference genome was downloaded from the 1000G_2504_high_coverage collection hosted at the IGSR FTP database (#link("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/")) in Compressed Reference-oriented Alignment Map (CRAM) format #c[Fairley 2020].
CRAM-based workflows require the GRCh38 reference sequence that was used during alignment; accordingly, the GRCh38 reference genome and the centromeric locations were downloaded from the IGSR FTP site.
An MD5-cache of the reference was built using the `seq_cache_populate.pl` script from samtools and an htslib reference cache was configured to speed up CRAM access, decoding, downsampling, and downstream conversion to BAM for tools that require BAM input (e.g., Delly).
Telomere, short arm, and heterochromatin region locations were downloaded from the UCSC hg38 genome annotation database (#link("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/")).

== Sequence File Preparation

The high-coverage sequences were subsampled using `samtools view --subsample {fraction} --subsample-seed {seed}` down to 6x, 4x, and 2x to simulate coverages typical in sequencing data retrieved from standard BGE pipelines.
Before subsampling, centromere, telomere, short arms, heterochromatin regions, and non-standard chromosomes (i.e., decoys and alternate contigs) were excluded from the original via `bedtools subtract` due to their poor mappability or incompatibility in comparison against the benchmark sets.
Each chromosome was subsampled individually to ensure even coverage.
The subsampling was done with a defined set of subsample seeds for reproducibility.

== Sequence-based CNV Calling <m_sequence_based_cnv_calling>

Three tools were used to call CNVs on sequencing data across all coverage types: CNVpytor #c[Suvakov 2021], GATK-gCNV #c[Babadi 2023], and Delly #c[Rausch 2012].
To ensure appropriate comparison and consensus merging across callers, each tool was set to use a 1000 bp bin size.

=== CNVpytor

CNVpytor was cloned from a GitHub repository (#link("https://github.com/abyzovlab/CNVpytor"), access date 03/12/2026) and installed, along with its dependencies, into a `pip` virtual environment.
The tool was run in a bash script to retrieve the read-depth, histograms, partitions, and calls for each sample at a 1000 bp bin size.
The cnvpytor interactive interface was automated to retrieve the final CNV calls in .vcf format.

=== GATK-gCNV

The GATK release 4.6.2.0 was downloaded from a GitHub repository (#link("https://github.com/broadinstitute/gatk")) and installed as a conda-env into a pixi environment.
The gCNV pipeline as outlined on the Broad Institute website was adapted to this study.
An interval size of 1000 bp was used, and the interval merging rule specified at all steps was set to "OVERLAPPING_ONLY".
Intervals across the genome were split into 24 equally sized shards to employ a scatter and gather methodology for more efficient runtime.
Once the pipeline was complete, identified CNVs were retrieved from the .vcf.gz files of the genotyped segments output.

=== Delly

The statically linked binary for Delly v.1.7.2 was downloaded from a GitHub repository (#link("https://github.com/dellytools/delly")) and run in a bash script.
CRAM sequence files were passed directly into the binary with the appropriate genome reference file.
The `delly cnv` command was used with the mappability map for GRCh38 hosted on the Delly FTP server to generate a .bcf file with the CNV calls.

=== Post-Processing

Some callers treat regions previously excluded in the preparation step (centromeres, telomeres, short arms, heterochromatin, decoy/alternate contigs) as structural variants.
Therefore, `bedtools intersect -v` was used to filter out CNVs that overlapped excluded regions by 50% of the interval length or greater to exclude the artifacts.

== SNP Array-based CNV Calling

SNP array data was downloaded from 1000 Genomes Project phase 3.
The samples were genotyped using the Illumina HumanOmni2.5-4 v1 DNA Analysis BeadChip.
Normalized genotype intensities were downloaded from the supporting Broad dataset from the IGSR FTP database (#link("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/broad_intensities/")).

PennCNV #c[Wang 2007] was used to perform CNV calling.
PennCNV uses a hidden Markov model to infer CNV calls.
Release v.1.0.5 of the tool was downloaded from a GitHub repository (#link("https://github.com/WGLab/PennCNV")) and the perl scripts included with the tool were used according to the "CNV calling" guidelines on the PennCNV website (#link("https://penncnv.openbioinformatics.org/en/latest/user-guide/test/")).

The signal intensity files for 2,141 samples were extracted from the Broad signal intensities file and formatted to be compatible with PennCNV.
A .pfb file was compiled using all samples' signal intensity files.
PennCNV was then used to detect CNVs and also filter out low quality CNVs, which was performed using `filter_cnv.pl` with the default settings specified by the tool.

== Benchmark Dataset Preparation <m_benchmark_prep>

The three benchmark datasets were chosen to serve as the benchmark sets for the CNVs in GRCh38.

Throughout this study a CNV is defined as a variant that changes the copy number of an interval of the reference assembly: a deletion (DEL), which lowers the copy number of a reference interval, or a duplication (DUP), which raises it.
The definition is imposed by the detection method under evaluation rather than chosen for convenience.
All three sequence-based callers infer copy number from sequencing depth over intervals of the reference, and the classification of a call as a true or false positive is decided by reciprocal overlap between two reference intervals.
A variant is therefore only assessable if it occupies an interval of the reference whose copy number differs from two.
Applying that criterion to the benchmark releases retains three record classes and excludes the rest.
Deletions are retained, including the mobile-element deletion classes that 1000 Genomes phase 3 names separately (`DEL_ALU`, `DEL_LINE1`, `DEL_SVA`, and `DEL_HERV`), as each removes a reference interval and differs from a plain deletion only in the annotation of the sequence removed.
Duplications are retained, including tandem and interspersed subclasses.
Multi-allelic copy-number records, whose alternate alleles are absolute copy numbers relative to a diploid reference (`<CN0>`, `<CN1>`, `<CN3>`, and so on), are resolved separately for each alternate allele, so that a carrier of a `<CN0>` allele contributes a deletion and a carrier of a `<CN3>` allele at the same record contributes a duplication; the `<CN2>` allele is the reference copy number and contributes nothing.

Three classes are excluded.
Insertions of novel sequence, whether unclassified (`INS`) or attributed to a mobile element (`ALU`, `LINE1`, `SVA`, `MEI`, `HERV`), are excluded because they occupy no interval of the reference: the reference span of such a record is either a single base or absent altogether, so overlap against a read-depth call is undefined and no depth-based caller can be scored against them.
This exclusion is consequential, since assembly-based variant representations of the kind used by HGSVC3 and ONT Vienna encode a tandem duplication as an insertion of the duplicated sequence at its own locus rather than as a copy-number gain over a reference interval, and the two cases are not separable from the released fields alone.
Inversions and breakends are excluded as copy-number neutral.
Records for which no end coordinate could be derived, from either the `END` or the `SVLEN` key of the `INFO` field, are excluded for want of a reference interval.

The 1000 Genomes phase 3 #c[1000G 2015] SV annotations were downloaded from the IGSR FTP site (#link("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/integrated_sv_map/supporting/GRCh38_positions/ALL.wgs.integrated_sv_map_v2_GRCh38.20130502.svs.genotypes.vcf.gz")) and contained 2504 samples.
This benchmark is derived from a combination of low-coverage WGS, high-coverage WES, and microarray genotyping #c[Byrska-Bishop 2021].
The SV set had positions that were originally identified on GRCh37 and lifted over to GRCh38 using the UCSC liftover tool, which consequently resulted in the removal of SVs that did not have a viable GRCh38 equivalent due to unmappability or significant size changes (>10%).

Human Genome Structural Variation Consortium, Phase 3 (HGSVC3) #c[Logsdon 2025] is an extensive analysis of 65 individuals of diverse ancestries across 27 distinct populations.
The benchmark contains SVs derived from PacBio HiFi long reads (\~47x coverage) and ultra-long Oxford Nanopore Technologies (ONT) reads (\~56x coverage).
The dataset contains annotations for variants derived from sequences natively aligned to GRCh38.
The GRCh38 SV InsDel Alt annotation file under HGSVC3 2024 v.1.0 was downloaded from the data collections on the IGSR FTP server (#link("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC3/release/Variant_Calls/1.0/GRCh38/variants_GRCh38_sv_insdel_alt_HGSVC2024v1.0.vcf.gz")).

The Oxford Nanopore Technology (ONT) Vienna #c[Schloissnig 2024] dataset includes long-read sequencing and SV characterization of 1,019 samples from the 1000 Genomes project.
The dataset contains annotations for variants derived from sequences aligned to GRCh38.
The SVIM HG38 annotation file under release v.1.1 was downloaded from the data collections on the IGSR FTP server (#link("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1KG_ONT_VIENNA/release/v1.1/svim-asm-hg38/svim.asm.hg38.bcf")).

== BED File Conversion <m_bed_file_conversion>

Sequenced-based CNV calls, SNP Array CNV calls, and benchmark CNV calls were all converted from their native formats into BED format using Python.
One BED file per sample was generated and contained the following information: chromosome number, start position, end position, the two structural variant types (DEL or DUP), and the source of the CNV call.

The VCF-like files were all processed using the cyvcf2 package #c[cyvcf2] package.
CNV end positions were extracted from the END key in the INFO field, or calculated from the SVLEN key in the INFO field when the former was not available.
Structural variant type (DEL or DUP) was extracted from the SVTYPE key in the INFO field, or was derived from the RCDN (read-depth copy number) key in the FORMAT field when the former was not available.

For the SNP Array CNV calls, the PennCNV final output file was split to retrieve per sample calls in the BED format, and the positions were lifted over from its native hg18 positions to hg38/GRCh38 using the Python liftover package and chain files from the UCSC hg38 database (#link("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/")).
CNVs with unmappable positions or those that changed in size by more than 10% were excluded, matching the restrictions from the liftover done for the 1000 Genomes phase 3 benchmark call set.

After liftover, any calls deemed artifactual due to overlapping with poorly mappable regions -- including centromeres, telomeres, and heterochromatin regions -- were removed from the call set.
An 1% overlap with a poorly mappable region was the criteria to remove a call, with other more and less stringent criteria also being tested for comparison.
The behavior of this filtering on the number of calls removed as well as the number of bases overlapping unmappable regions, the number of bases removed in total, and the ratio between the two were plotted and analyzed.

== CNV Overlap and Adjacency Graph Building <m_cnv_overlap>

After retrieving all CNV call sets of interest and converting them to a standardized BED format, the CNV data was loaded into Python to generate a graph representation of call sets.
Individual graphs were generated for 1) All benchmark calls, 2) Sequence-based CNV calls per tool, 3) Sequence-based CNV calls aggregated across all tools, and 4) SNP array CNV calls.

The edge calculation process began by generating an iterable list of nodes and ensuring that the input is chromosome and start sorted.
An empty connected component is initialized for each unique partition of tuple (Sample ID, Chromosome, SV type), defining the boundary across which edges cannot be made.
For each sequential node, it checks whether or not there exists overlap between itself and any nodes in the appropriate partition's connected component: If there is, create an "overlap edge" between nodes with the reciprocal overlap as the weight; otherwise, create a "gap edge" between the nodes with their distance from each other as the weight and reset the connected component to only contain the current node.
This continues until all edges are built.

== Consensus CNV Calling <m_consensus_calling>

Merging sub-networks of the graphs according to different edge and network filtering parameters was also performed in order to derive consensus CNV call sets.
Edges were filtered by minimum reciprocal overlap threshold or by maximum distance between calls, in a mutually exclusive manner.
Once the edges were filtered, all connected components were retrieved and additionally filtered for a minimum number of unique sources across the nodes in each component.
Once all connected components that passed the filters were retrieved, the nodes within each component were aggregated by union into a single child call node, taking the minimum start position and the maximum end position across all member nodes.

Consensus CNV call graphs were generated for the benchmark sets to create a single merged benchmark set to serve as the primary truth source for downstream analysis.
Consensus call graphs were also generated for the three sequence-based call sets derived from the calling tools of interest.
Multiple versions of these two consensus call graphs were generated across different filtering parameters for downstream analysis of performance across the parameter field.

== Null Model for Caller Agreement <m_null_model>

Consensus construction assigns every merged component to one of seven categories according to which of the three callers contributed a call to it.
Those categories describe how the callers relate to one another, but on their own they do not distinguish callers that detect a common population of events at different rates from callers that detect different populations.
To separate the two, we fit a null model in which a single population is assumed and identify where the model fails.

Let $N$ be the number of events available to be detected, and let $p_i$ be the probability that caller $i$ detects any one of them, independently of the other two callers.
Writing $n_A$ for the number of components detected by exactly the set of callers $A$, the model's expectation for each category is

$ EE [n_A] = N product_(i in A) p_i product_(i in.not A) (1 - p_i) $

The model has four parameters and the seven categories supply seven counts, so it is over-determined and can be fit on a subset of them.
We fit it on the four categories in which at least two callers agree, which is the part of the data a single-population model is meant to describe.
Writing $n_"all"$ for the all-three category and $n_(-i)$ for the pairwise category from which caller $i$ is absent, the two differ only in whether caller $i$ detected the event, so their ratio removes $N$ and both of the remaining rates,

$ (EE [n_"all"]) / (EE [n_(-i)]) = p_i / (1 - p_i) $

Each rate is therefore determined by a single observed ratio $r_i = n_"all" \/ n_(-i)$, and the pool size follows from the all-three category alone.

$ hat(p)_i = r_i / (1 + r_i) wide hat(N) = n_"all" \/ product_i hat(p)_i $

The three single-caller categories take no part in the fit.
They are predicted rather than described, which leaves the model three degrees of freedom against which it can be rejected, and it is the size of the discrepancy in those three categories that carries the result.
Category sizes, median interval sizes, and duplication shares were computed from the same merged components used for consensus call set construction, at 30x and at a 50% reciprocal overlap threshold.

== Binary Classification

CNV calls derived from either the sequence-based CNV calling tools or the SNP array, including both the raw calls from the tools and the consensus calls across the sequence-based tools, were defined as query call sets.
These call sets were run against the truth call sets derived from the consensus of the three benchmark call sets.

Query and truth calls were used to generate a separate graph with overlap edges being calculated in the same manner as the consensus calling.
Edges were then filtered down based on reciprocal overlap threshold.
Binary classification of the calls were then performed on the filtered graph.

Query CNV calls were marked as true positive if there existed an edge to a truth CNV call, or marked as false positive otherwise.
Truth CNV calls were marked as "found" if there existed an edge to a query CNV call, or false negative otherwise.
From the counts of these metrics, the Precision, Recall/Sensitivity, and $F_beta$-score were derived using the formulas listed below.

#v(0.4em)
#align(center)[
  $ "Precision" = "TP" / ("TP" + "FP") wide "Recall" = "TP" / ("TP" + "FN") $
  $ F_beta = (beta^2 + 1) / ((beta^2 dot "recall"^(-1)) + "precision"^(-1)) $
]
#v(0.4em)

A beta value of 1 was used for the F-score.
CNV calls present in the benchmark set that were not identified in any sample in any of the CNV call sets were excluded from the false negatives for the binary classification, as these were deemed as undiscoverable by the data and calling algorithms that we used.

Figures were generated with matplotlib \[v3.11.1\].
Two families of plots summarize the classification.
Size-resolved performance is shown as kernel-smoothed density curves over CNV size, together with cumulative and complementary cumulative distribution functions giving the proportion of CNVs at or below, and at or above, each size threshold.
Query-to-truth matching structure is summarized by the number of query calls matching more than one truth call and vice versa; because matching is many-to-many, true-positive counts on the query and truth sides do not necessarily coincide.

== Parameter Selection

Four parameters govern the comparison between the query and truth call sets.
The benchmark padding decides when records from different benchmark sets describe the same event based on their proximity to each other.
The size floor decides the smallest event the comparison is allowed to score, and therefore which intervals enter it from the query or truth side.
The query consensus reciprocal-overlap threshold decides when calls from different callers describe the same event, and affects which calls from the sequence-based caller outputs propagate into the consensus call sets.
The classification reciprocal-overlap threshold decides when a query call and a benchmark interval describe the same event.

Each parameter was then profiled over its range with the other three held at the values adopted for the pipeline.
Precision, recall, and F1 were recorded at every point, together with the number of intervals entering the comparison on each side and the match topology, so that a change in a metric can be attributed to a change in total set size rather than reported as a change in performance.

=== Benchmark padding

Padding is applied to both ends of every interval, so a padding of $p$ increases an interval's span by $2p$ and bridges any two intervals separated by no more than $p$.
Its intended role is to absorb the breakpoint imprecision between benchmark sets built from different technologies and assemblies, and it carries that meaning only while $p$ remains small relative to the intervals it acts on.
Once $p$ approaches their typical span, bridging no longer joins two descriptions of one variant and begins fusing descriptions of distinct ones.
A second consequence follows from the order of operations, the size floor being applied after merging rather than before it: fusion can carry a run of sub-floor fragments across the floor as a single interval.
This is intended to aggregate these runs of near-adjacent fragments into larger intervals such as those sequence-based callers with limited, short read-based evidence are more likely to identify.
Padding was profiled over \[0, 100 kb\], a range wide enough to reach the regime in which it dominates the intervals it is applied to, and the transitions are reported with the results.
For subsequent performance comparisons, padding was capped at 1 kb; this is the bin size common to all three callers and therefore the largest boundary error the comparison can be asked to tolerate.

=== Size floor

All three sequence-based callers were run with a 1 kb bin size and consequently cannot resolve a CNV narrower than that, whereas the merged benchmark is dominated by events an order of magnitude smaller.
Comparing the two sets without a size restriction therefore results in callers failing to detect variants that the chosen bin size places outside their resolution, which reduces recall for a reason that has nothing to do with sequencing depth.

The floor is applied after both call sets have been merged rather than before, which is what allows the padding of the preceding subsection to carry a run of sub-floor benchmark records across it.
It was applied to the query and truth call sets symmetrically, both sides being restricted to intervals of at least the floor before any matching was performed.

The floor was profiled over \[1 bp, 100 kb\] at 80 logarithmically spaced points for each of the six 30x query call sets, and takes the values \[0, 250, 500, 1000, 2000, 5000, 10000\] within the joint grid.
Alongside the metrics we tracked the number of intervals surviving in each call set and the maximum attainable recall, defined as the ratio of query calls to truth intervals and capped at one.
That ratio is the largest recall achievable if every query call were to match a distinct truth interval, so it bounds recall from above independently of how well the callers perform, and it identifies the regime in which a call set has become larger than the truth set it is being scored against.

=== Query consensus reciprocal overlap

A consensus component is the transitive closure of the pairwise overlaps that pass this threshold, so the threshold dictates what degree of agreement between calls is required for calls to propagate.
At the permissive end, where a single shared base pair is sufficient to record two calls as equivalent and merge them, transitivity admits of chains of calls that each overlap a neighbor without overlapping one another, and a component's span can then exceed that of any call within it; where that happens the consensus level records co-location rather than agreement about an event.
Whether it happens is a property of the call sets, so every component was summarized by the ratio of its span to the span of the longest single call within it, with a ratio of one indicating a component whose extent is set by a call some caller actually reported.
At the stringent end only calls with nearly coincident coordinates merge, so two callers that detect the same variant but place its breakpoints by different conventions are recorded as disagreeing.
The threshold was examined over \[0.05--0.95\] in steps of 0.05.

=== Classification reciprocal overlap

The same geometry applies between a query call and a benchmark interval, with one consequence specific to matching.
At a threshold of 0.5 or above a query call cannot cover half of each of two benchmark intervals unless those intervals themselves overlap; below 0.5 the matching becomes genuinely many-to-many, and a single call can be credited against several benchmark intervals at once.
A benchmark merged at zero padding is disjoint within every sample, chromosome and variant type by construction, so above 0.5 no query call can be credited twice; a query set merged on reciprocal overlap is not, since two components can overlap one another below the threshold at which they were built, so a benchmark interval may still be split between query calls.
Match topology was therefore recorded on both sides alongside the metrics, since it is what distinguishes a threshold that admits more correct matches from one that admits the same match repeatedly.
A threshold of exactly zero admits a single shared base pair as a match; it is retained in the profile, where it bounds the topology, and excluded from the joint grid.
The threshold was profiled over \[0--0.99\] in steps of 0.01.

== Variance-based sensitivity analysis and Pareto front <m_vbsa_pf>

The profiles above vary one parameter at a time.
They describe the pipeline completely only if the effect of moving one parameter does not depend on where the other three are held, which is an assumption about the shape of the metric field rather than something the profiles themselves can show.
All four parameters were therefore also varied jointly, over the full factorial grid of the ranges fixed above with padding taking \[0, 10, 25, 50, 100, 200, 400, 700, 1000\], the size floor taking \[0, 250, 500, 1000, 2000, 5000, 10000\], and both reciprocal-overlap thresholds taking \[0.05--0.95\] in steps of 0.05, and precision, recall, and F1 were evaluated at every combination together with the number of query and truth intervals entering it.
What follows asks how much of the variation in each metric the one-at-a-time reading accounts for, and where a joint reading is required instead.

=== Sensitivity Indices

The contribution of each parameter was quantified by variance-based sensitivity analysis (Sobol', 1993).
Writing $Y$ for a metric (precision, recall, f1) and $x_1, ..., x_4$ for the parameters, the first order Sobol index $S_i$ of parameter $i$ is the fraction of the metric's variance attributable to that parameter acting alone.

$ S_i = ("Var" (EE [Y | X_i])) / ("Var" (Y)) $

The total-order index $S_(T i)$ additionally includes every interaction involving $i$.

$ S_(T i) = 1 - ("Var" (EE [Y | bold(X)_(tilde i)])) / ("Var" (Y)) $

The difference $S_(T i) - S_i$ is the variance a parameter contributes only jointly with others.
Because the design is a complete factorial grid, each conditional expectation is a marginal mean over the grid and both indices were computed exactly.
The complete decomposition, comprising every term up to fourth order, sums to one, and this was verified numerically.

=== Additivity

The same decomposition expresses the metric field as a grand mean plus one univariate function per parameter:

$ Y approx mu + sum_i f_i (X_i) wide "with" wide f_i (x) = EE [Y | X_i = x] - mu $

The coefficient of determination of this additive model equals the sum of the first-order indices, and so quantifies directly the extent to which the one-at-a-time profiles of the preceding section describe the joint behavior of the pipeline.

=== Dependence on the swept ranges

Sobol indices are variance ratios with respect to a distribution over the inputs.
They therefore describe sensitivity within the examined region and are not invariant to the choice of that region.
As such, indices were additionally computed over an intentionally over-wide grid extending to $10^6$ bp (Supplementary Table X) in order to compare physical arguments for the previously described parameter boundaries with empirical data.

=== Pareto front

Because precision and recall span very different ranges across the grid, F1 is close to a monotone function of recall alone and obscures the trade-off between the two.
Parameter settings were therefore also summarized by their Pareto front: a setting is dominated if another attains at least equal precision and recall and strictly exceeds it on one, and the front comprises the non-dominated settings.
The front answers a different question from an optimum.
It separates settings that buy precision at a real cost in recall from settings that are simply worse on both, so a dominated setting has no argument in its favor.
Dominance compares measured performance, so it is informative only between settings that score the same events under the same definition of a match.
Three of the four parameters do not satisfy that: the padding and the floor change which intervals enter the comparison, and the classification threshold changes what counts as a match, so for these the front records a change in the comparison rather than an improvement in the pipeline.
Fronts were therefore computed over the whole grid and again with the classification threshold, and then the size floor, held at the values adopted for the pipeline.

=== Implementation

Analyses were performed in Python \[3.14.6\] with NumPy \[2.5.1\] and SciPy \[1.18.0\].

= Results

== CNV Calling Overview

#figplaceholder("image7 — Figure 1, overall analysis workflow", height: 7cm)

#v(0.6em)

To evaluate the performance of CNV detection with BGE-compatible lcWGS relative to SNP microarrays, we benchmarked CNV call sets across thirteen 1000 Genomes Project individuals selected based on the availability of high-coverage WGS data, microarray data, and three available callsets.
The samples cover a diverse range of continental superpopulations (Supplemental Table 1000G Populations) and contain seven males and six females.

For CNV calling tool selection, we reviewed several sequencing-based CNV calling tools (Supplemental Table Tools) and selected CNVpytor, GATK-gCNV, and Delly for testing.
CNVpytor #c[Suvakov 2021] is a successor to CNVnator that performs read-depth based CNV detection.
In addition to read-depth, CNVpytor allows for the addition of SNPs and small indels for additional evidence in CNV calling; this functionality was purposefully left unused to test the caller's performance in a scenario closer to that of a study with only BGE-derived lcWGS data available.
GATK-gCNV #c[Babadi 2023] is a pipeline composed of various GATK functions and algorithms to perform germline CNV calling.
Many of the functions were originally from the Picard toolkit #c[Broad Institute 2019], and have since been integrated into the GATK toolkit.
This pipeline gathers read-depth based information, clusters samples with similar read-depth profiles via principal component analysis (PCA), and builds a unified probabilistic model to perform read-depth denoising and CNV inference.
Delly #c[Rausch 2012] is a program that uses read-depth information, paired-end mapping, and split-read analysis to perform CNV calling.
It combines short-range and long-range information and uses GC and mappability fragment correction to identify CNVs.
Compared to CNVpytor and GATK-gCNV which are limited by only using read-depth methodologies, Delly can infer base-level SV breakpoints within bins.

The overall workflow of the analysis is outlined in Figure 1.
Briefly, we first subsampled the high-coverage (30x) WGS data of the thirteen samples of interest to 6x, 4x, and 2x to simulate coverages typically retrieved from BGE pipelines.
We then used three different tools, CNVpytor #c[Suvakov 2021], GATK-gCNV #c[Babadi 2023], and Delly #c[Rausch 2012], to call CNVs across all coverage types.
Next, we generated consensus CNV callsets by combining CNVs that were of the same CNV type (deletion vs. duplications) and were identified in different numbers of calling tool outputs.
We additionally retrieved benchmark sets for the thirteen samples being analyzed from the 1000 Genomes phase 3, HGSVC3, and ONT Vienna SV sets and took a union of those sets to generate our truth benchmark set.
Using the 30x data as a reference, we then characterized how the performance responded to the size floor and to the consensus merging parameters.
We confirmed that this response held across all four coverages and across every individual consensus call set, then fixed a single parameter set on physical and empirical grounds and applied it to all coverages to improve downstream analysis and interpretability.
Finally, we compared the individual tool call sets, the consensus call sets, and the SNP-array control call set against the benchmark set to evaluate the performance of the lcWGS for CNV calling.

== Input Call Sets After Parsing <r_input_call_sets>

All input call sets were normalized to a common per-sample BED representation and every call set was restricted to the thirteen samples of interest to ensure no statistic reflected differences in cohort composition.
Liftover was performed on the calls derived from the SNP Array from hg18 to hg38/GRCh38 to align coordinates with all other call sets.
Table 1 summarizes the resulting call sets: the three sequence-based callers at each of the four coverages, the SNP array control, and each of the three benchmark sources.
Counts are given after removal of calls overlapping poorly mappable regions by at least 1% of their total length.
Pre-filter counts and the full liftover accounting are given in Supplementary Table S1, and the complete exclusion accounting -- bases removed against bases actually inside the mask, and the split by CNV type -- in Supplementary Table S2.
The results from different stringencies for overlap with poorly mappable regions were tested and are given in Supplementary Table S3.
1% was determined to be the most reasonable choice for downstream analysis on the physical grounds that increasing overlap percentage would permit more calls with unreliable breakpoints and decreasing overlap requirements would more often cause true calls to be removed; empirical results validated this.

#v(0.5em)

#block(width: 100%)[
#set text(hyphenate: false, size: 10pt)
#table(
  columns: (1fr, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right),
  table.header(
    [Call Set], [Calls], [Median /\ Sample], [MAD], [% DEL],
    [Median\ Size (bp)], [% Masked],
  ),
  [*30x -- CNVpytor*], [16,159], [1,238], [24], [76.47], [14,000], [45.37],
  [*30x -- Delly*], [7,025], [539], [32], [71.90], [4,112], [5.74],
  [*30x -- GATK-gCNV*], [8,216], [610], [80], [89.12], [3,000], [40.44],
  table.hline(stroke: 0.3pt),
  [*6x -- CNVpytor*], [8,480], [652], [22], [62.70], [40,000], [9.08],
  [*6x -- Delly*], [4,107], [314], [8], [47.77], [4,718], [5.02],
  [*6x -- GATK-gCNV*], [8,511], [666], [29], [85.94], [4,000], [0.27],
  table.hline(stroke: 0.3pt),
  [*4x -- CNVpytor*], [7,300], [562], [13], [59.21], [49,000], [10.36],
  [*4x -- Delly*], [3,739], [287], [16], [41.96], [4,735], [2.63],
  [*4x -- GATK-gCNV*], [5,035], [386], [22], [62.80], [5,000], [0.34],
  table.hline(stroke: 0.3pt),
  [*2x -- CNVpytor*], [6,454], [498], [8], [52.68], [51,000], [8.84],
  [*2x -- Delly*], [3,040], [238], [6], [31.97], [4,588], [3.12],
  [*2x -- GATK-gCNV*], [10,553], [839], [11], [56.95], [6,000], [0.08],
  table.hline(stroke: 0.3pt),
  [*SNP Array*], [1,659], [111], [19], [52.56], [7,554], [0.00],
  table.hline(stroke: 0.3pt),
  [*1000G phase 3*], [39,029], [3,536], [452], [98.98], [615], [0.00],
  [*HGSVC3*], [172,922], [13,441], [813], [100.00], [146], [0.61],
  [*ONT Vienna*], [105,089], [7,672], [152], [100.00], [118], [2.44],
)
]

#cap("Table 1:")[
  Parsed input call sets after removal of calls overlapping poorly mappable regions.
  Every call set covers the same thirteen 1000 Genomes samples.
  Median per Sample and MAD describe the per-sample call count; MAD is the median absolute deviation about that median.
  Removed by Mask is the percentage of parsed calls discarded for overlapping a centromere, telomere, short arm, heterochromatin region, or decoy/alternate contig by more than 1% of their length.
  Pre-filter counts appear in Supplementary Table S1 and the full exclusion accounting in Supplementary Table S2.
]

#v(0.8em)

The three callers differ markedly in both yield and size regime, and the differences are not stable across coverage.
CNVpytor produced the largest call set at 30x (16,159 calls) and its yield fell monotonically with coverage, reaching 6,454 calls at 2x.
Delly followed the same direction over a smaller range.
GATK-gCNV did not: it produced more calls at 2x (10,553) than at any other coverage including 30x (8,216), which is consistent with a caller that lacks a mechanism for withholding calls when the underlying read-depth evidence becomes too sparse to support them.
Median call size moved in the opposite direction to yield for the read-depth callers, with CNVpytor rising from 14 kb at 30x to 51 kb at 2x; Delly, which refines breakpoints from split reads rather than bin boundaries, held a median near 4.6 kb at every coverage.
The proportion of deletion calls also fell with coverage for every caller, most sharply for Delly, which went from 71.9% deletions at 30x to 32.0% at 2x.

#v(0.3em)

// TODO(lionel): confirm against the subsampling scripts before this goes in the
// submitted draft. If the masking asymmetry is a pipeline-stage artifact rather
// than a coverage effect, this paragraph should become a Methods correction.
The fraction of calls removed by the mappability filter separates the 30x arm from the three low-coverage arms.
At 30x, 45.4% of CNVpytor calls and 40.4% of GATK-gCNV calls overlapped an excluded region, against 8.8-10.4% and 0.1-0.3% respectively at 2x, 4x, and 6x.
This is expected from the order of operations rather than from coverage itself: excluded regions were subtracted from the alignments before subsampling, so the low-coverage inputs contained no reads over those regions and the callers could not emit calls there, whereas the 30x arm was called from the original alignments and the same regions were removed only afterwards, at the level of calls.
The call sets that enter the downstream analysis are equivalent in that both have had these regions removed, but the read-depth normalization performed internally by CNVpytor and GATK-gCNV was not carried out over the same genomic territory in the two cases.
Delly, which applies its own mappability map during calling, is the least affected caller at every coverage.

#v(0.3em)

Most of the content of the three benchmark releases is not a copy-number change, and applying the inclusion criteria of the Methods excluded 17,573 records from 1000 Genomes phase 3 (16,787 insertions and 786 inversions), 110,623 from HGSVC3, and 89,257 from ONT Vienna, the latter two consisting entirely of insertions.
The retained records yielded 39,029, 172,922, and 105,089 calls respectively across the thirteen samples, making the benchmark sources the largest of the parsed call sets.

#v(0.3em)

The benchmark is dominated by events below the resolution of a 1 kb read-depth bin.
Merging the three sources into a single truth set, with the zero-padding settings used on the truth side of every comparison below, yielded 178,838 intervals with a median size of 145 bp, of which 13.9% reach 1 kb and 2.8% reach 10 kb.
Composition differs sharply by source: 40.7% of 1000 Genomes phase 3 intervals reach 1 kb, against 12.2% for HGSVC3 and 8.6% for ONT Vienna.
The two assembly-based sources therefore supply most of the intervals but little of the mass in the size range the callers operate in, and the sub-1 kb remainder is beyond the reach of a read-depth call at any reciprocal-overlap threshold.
This is the principal reason a size floor is required before recall can be interpreted at all, and it is taken up below.

#v(0.3em)

// TODO(lionel): this paragraph is written for the benchmark trio as parsed. If
// the SVAN annotation on the ONT Vienna release is used to recover tandem
// duplications, the counts and the final two sentences change.
The composition of the merged truth set by CNV type constrains the rest of the analysis.
It is 99.8% deletions: only 343 of its 178,838 intervals are duplications, and all 343 come from the multi-allelic copy-number records of 1000 Genomes phase 3.
HGSVC3 and ONT Vienna contribute none.
This reflects how those two releases represent variation rather than the populations they describe, since an assembly-based caller encodes a tandem duplication as an insertion at its own locus, which occupies no reference interval.
Above a 1 kb floor the imbalance is unchanged, with 343 duplications among 24,820 intervals.
Duplication-specific recall therefore cannot be estimated against this benchmark trio.
All results below are reported over deletions and duplications combined and are in practice a measurement of deletion performance; duplication calls in the query sets are scored against a truth set that holds almost no duplication content, so the false positives they generate are not evidence that those calls are wrong.

#v(0.3em)

The SNP array control contributed 1,659 calls across the thirteen samples, the smallest of the evaluated call sets, with a median size of 7,554 bp and a near-even split between deletions and duplications (52.6% deletions).
No array call overlapped an excluded region, which is expected given that the array's probes are not sited in the regions the mask covers.
Sixty-one array calls (3.5%) were lost during liftover from hg18 to hg38, 18 because an endpoint failed to map and 43 because the interval changed length by more than 10% (Supplementary Table S1).

== Sequence-based Consensus Call Set Construction <r_sequence_consensus>

All outputs from the sequence-based consensus call sets were aggregated into a single graph and edges between overlapping and adjacent calls were computed accordingly.
Analysis of the consensus construction was required in order to determine the general behavior of each of the callers and if consensus analysis was a suitable approach to retrieve an informative population of calls.
We therefore characterized the three callers relative to each other at 30x, where the evidence available to them is greatest, and then followed the same quantities down through the reduced coverages.

Agreement between the callers is significant, but it is not distributed evenly across the multi-caller categories (Figure 2).
Of the 24,123 components in the 30x 1/3 consensus call set, 19,287 (79.9%) carry a single caller, and CNVpytor alone accounts for 11,952 of them, just under half of the call set.
Of the 4,836 components carrying at least two callers, the three pairwise-only categories hold 939, 868, and 629 components, while agreement between all three callers (3/3) holds 2,400.
A component found by more than one caller is therefore about as likely to have been found by all three as by exactly two.

#figure(
  image("/results/consensus/caller_agreement_30x.png", width: 100%)
)

#cap("Figure 2:")[
  Venn diagram of source distribution for CNV call components.
  Components were identified from 30x coverage WGS data by the following sequence-based CNV calling tools: CNVpytor, GATK-gCNV, and Delly.
  The CNV components came from a 1/3 consensus call set generated with a 50% reciprocal overlap requirement between CNVs across tool outputs.
  The total counts per caller slightly deviate from the original raw counts seen in Table 1; the aggregation of calls with one-to-many overlaps across tool outputs was required to prevent double-counting and is what causes the slight count disparity.
  The size of each partition in the graph is not to scale, with a prioritization on readability.
]

#v(0.8em)

The categories differ in what they contain as well as in how large they are (Table 2).
Components carried by CNVpytor alone have a median size of 21,000 bp, an order of magnitude above the 2,000 bp of components carried by GATK-gCNV alone and the 2,148 bp of those carried by Delly alone, and well above the 6,186 bp of the components that all three callers share.
Their spread differs by as much again, with a median absolute deviation of 17,000 bp for CNVpytor-only components against 1,000 bp for GATK-gCNV-only components.
Duplications are the larger population in every category, exceeding deletions in median size by factors of 1.5 to 12.9, so the two directions are reported separately.
The duplication share falls as agreement rises, from 28.7% of CNVpytor-only components to 4.8% of those found by all three, and the categories containing Delly carry the largest duplication shares at every level of agreement.

#v(0.5em)

#block(width: 100%)[
#set text(hyphenate: false, size: 9pt)
#table(
  columns: (1fr, auto, auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right, right, right, right),
  table.header(
    table.cell(rowspan: 2)[Agreement Category],
    table.cell(rowspan: 2)[Compo-\ nents],
    table.cell(colspan: 3)[Median Size (bp)],
    table.cell(colspan: 3)[MAD Size (bp)],
    table.cell(colspan: 2)[Composition],
    [All], [DEL], [DUP], [All], [DEL], [DUP], [% DEL], [% DUP],
  ),
  [*All three callers*], [2,400], [6,186], [6,028], [19,174], [2,186], [2,028], [8,174], [95.2], [4.8],
  [*CNVpytor + GATK-gCNV*], [939], [4,000], [4,000], [22,500], [1,000], [1,000], [8,000], [95.7], [4.3],
  [*CNVpytor + Delly*], [868], [8,359], [4,752], [61,138], [5,699], [2,126], [33,614], [74.7], [25.3],
  [*Delly + GATK-gCNV*], [629], [3,104], [2,910], [6,000], [1,150], [958], [1,867], [83.9], [16.1],
  table.hline(stroke: 0.3pt),
  [*CNVpytor only*], [11,952], [21,000], [12,000], [33,000], [17,000], [10,000], [13,000], [71.3], [28.7],
  [*GATK-gCNV only*], [4,208], [2,000], [2,000], [3,000], [1,000], [1,000], [2,000], [84.9], [15.1],
  [*Delly only*], [3,127], [2,148], [1,898], [3,402], [1,050], [752], [2,304], [50.8], [49.2],
)
]

#cap("Table 2:")[
  Size and composition of the caller-agreement categories in the 30x 1/3 consensus call set.
  Categories in which at least two callers agree are given above the rule and single-caller categories below it.
  Median and MAD are taken over component sizes, MAD being the median absolute deviation about the median, and are reported over all components in a category and separately for its deletions and duplications.
  Deletion and duplication shares are complementary, as no other CNV type is represented in these call sets.
]

#v(0.8em)

Whether these categories are consistent with the callers detecting one population of events can be tested directly.
Fitting the null model of the Methods to the four categories in which at least two callers agree fixes the population at 5,738 events and the per-caller detection rates at 0.79, 0.73, and 0.72 for CNVpytor, GATK-gCNV, and Delly respectively.
The three single-caller categories take no part in the fit and are therefore predicted, and the model predicts 813 components across all of them against the 19,287 total that is observed (Table 3).
The whole visible output of the fitted model is 5,649 components, against the 24,123 the call set contains.
Caller agreement at 30x is therefore better described by two populations than by one: a core of events that every caller reaches at a comparable rate, and a caller-private component an order of magnitude larger whose extent is set by the assumptions of the individual caller.
Whether that private component consists of artifacts or of genuine calls beyond the reach of the other callers cannot be settled from agreement alone, and is taken up in the comparison against the benchmark.
The 3/3 category, although limited compared to the full aggregate call set, is nonetheless the subset with the highest discoverability across algorithms and may indicate the calls with the highest probability of being genuine CNVs.

#v(0.5em)

#block(width: 100%)[
#set text(hyphenate: false, size: 10pt)
#table(
  columns: (1fr, auto, auto, auto),
  align: (left, right, right, right),
  table.header(
    [Agreement Category], [Components], [Predicted], [Excess],
  ),
  [*All three callers*], [2,400], [2,400], [--],
  [*CNVpytor + GATK-gCNV*], [939], [939], [--],
  [*CNVpytor + Delly*], [868], [868], [--],
  [*Delly + GATK-gCNV*], [629], [629], [--],
  table.hline(stroke: 0.3pt),
  [*CNVpytor only*], [11,952], [340], [11,612],
  [*GATK-gCNV only*], [4,208], [246], [3,962],
  [*Delly only*], [3,127], [227], [2,900],
  table.hline(stroke: 0.3pt),
  [*Total*], [24,123], [5,649], [18,474],
)
]

#cap("Table 3:")[
  Observed caller-agreement categories against the counts predicted by the single-population null model.
  The model's four parameters were fixed on the four categories above the rule, which therefore reproduce their observed counts exactly and carry no excess.
  The three single-caller categories below the rule are predicted rather than fitted, and are where the model is tested.
  Fitted detection rates were 0.79 for CNVpytor, 0.72 for Delly, and 0.73 for GATK-gCNV, over an implied population of 5,738 events; the predicted total of 5,649 is smaller than that population because a further 89 events are expected to escape all three callers and so never appear as components.
]

#v(0.8em)

Extending the same construction to 6x, 4x, and 2x, the agreement requirement separates the coverages more sharply than the individual caller yields do.
The 1/3 call set inherits the non-linear coverage-to-call count relationship in GATK-gCNV, holding 19,146 components at 2x against 14,646 at 4x, whereas the 2/3 and 3/3 call sets fall monotonically with coverage, from 4,836 to 721 and from 2,400 to 177 components respectively (Supplemental Table Callsets).
The fraction of the call set that survives the requirement of higher consensus falls with coverage throughout: requiring two callers removes 80.0% of the 30x call set and 96.2% of the 2x call set, and requiring all three removes a further 50.4% and 75.5% respectively.
The 2x 1/3 call set is therefore both the second largest of the four and the one that loses the most to any agreement requirement, which suggests that much of it consists of caller-specific artifacts arising from reduced confidence at low coverage rather than of calls that the remaining callers failed to reach.

#v(0.3em)

Refitting the null model at each coverage shows that the two populations respond to depth in opposite ways (Table 4).
For the concordant population, the fitted core's total events (predicted number of calls given the null model holds) falls from 5,738 at 30x to 1,436 at 2x, and the detection rates per caller fall alongside it, from 0.79 to 0.59 for CNVpytor, from 0.72 to 0.45 for Delly, and from 0.73 to 0.46 for GATK-gCNV.
The two contribute in similar measure, since holding the rates at their 30x values while shrinking the core to its 2x size would leave 1,210 concordant components against the 721 observed.
The caller-private population does not follow.
The ratio of private to concordant components consequently rises as coverage decreases, from 4.0 at 30x to 25.6 at 2x.
This trend holds despite the total number of private calls not demonstrating a clear trend in depth, with call count being larger at 2x than at 4x.
This is evidence to support the proposed mechanism behind the losses to the agreement requirement described above: what additional depth supplies is primarily agreement between callers, while private calls from a caller are produced about as readily at 2x as at 30x.
Whether that private population is artifactual cannot be settled from its coverage response alone, although a population whose size is largely independent of the evidence available to produce it is difficult to attribute to the underlying genetic data.
The per-category counts behind this fit at each of the reduced coverages are given in Supplemental Table Agreement Categories.

#v(0.5em)

#block(width: 100%)[
#set text(hyphenate: false, size: 9pt)
#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, center, right, right, right),
  table.header(
    table.cell(rowspan: 2)[Coverage],
    table.cell(colspan: 3)[Components],
    table.cell(colspan: 2)[Fitted Core],
    table.cell(colspan: 3)[Detection Rate],
    [Concordant], [Private], [Ratio], [Events], [95% CI],
    [CNVpytor], [Delly], [GATK-gCNV],
  ),
  [*30x*], [4,836], [19,287], [4.0], [5,738], [5,571--5,904], [0.79], [0.72], [0.73],
  [*6x*], [1,708], [16,993], [9.9], [2,276], [2,150--2,411], [0.72], [0.61], [0.69],
  [*4x*], [1,119], [13,527], [12.1], [1,956], [1,775--2,174], [0.64], [0.57], [0.43],
  [*2x*], [721], [18,425], [25.6], [1,436], [1,251--1,675], [0.59], [0.45], [0.46],
)
]

#cap("Table 4:")[
  The two populations of caller agreement, fit separately at each coverage.
  Concordant components are those carrying at least two callers and private components those carrying one; Ratio is the second divided by the first.
  Fitted Core is the population of events implied by the null model, and Detection Rate the probability that each caller recovers one of them.
  Intervals are the 2.5th and 97.5th percentiles of 2,000 multinomial resamples of the seven category counts, with the model refit on each draw.
]

#v(0.8em)

#v(0.3em)

Composition moves with the agreement requirement as well as with call count.
Within each coverage, raising the requirement lowers the duplication share of the call set, but by an amount that shrinks as coverage falls: at 30x the share falls from 25.2% at 1/3 to 4.8% at 3/3, while at 2x it moves only from 48.3% to 44.6%.
Median component size moves in the opposite direction, rising within the 3/3 call sets from 6,186 bp at 30x to 55,487 bp at 2x.
The calls that survive the strictest agreement requirement at low coverage are consequently few, large, and no longer preferentially deletions, and at 177 components the 2x 3/3 call set is too small to support a finer breakdown than that.


== Parameterizing the Comparison <r_parameterizing>

Four user-defined parameters govern the comparison: the padding applied to the benchmark records before the three benchmark sets are merged, the size floor below which no event is scored on either side, the reciprocal-overlap threshold at which calls from different callers are merged into a consensus call, and the reciprocal-overlap threshold at which a query call is credited against a benchmark interval.
Each was bounded on geometric grounds in the Methods.
What follows reports what each one does to the comparison across its range, taking them in the order the pipeline applies them and holding the other three at their adopted values throughout: zero padding, a 1 kb floor, and a reciprocal overlap of 0.5 for both consensus construction and classification.

=== Benchmark Padding

The merged benchmark is built from three sets produced by different technologies and assemblies, so their breakpoints for the same variant do not coincide exactly, and padding is a standard remedy for that.
The merged benchmark has a median interval of 145 bp (IQR \[74--342\] bp), so padding of even a few hundred base pairs is comparable in size to the intervals it is meant to reconcile.
We swept padding from 0 to 100 kb and re-evaluated the comparison at every point (Figure 4).

#figure(
  image("/results/parameterization/benchmark_padding.png", width: 100%)
)

#cap("Figure 4:")[
  Effect of benchmark padding on the truth set and on the comparison, swept from 0 to 100 kb.
  Padding is applied to both ends of every benchmark record before the three benchmark sets are merged, and the 1 kb size floor is applied afterwards.
  (A) Benchmark intervals: all merged intervals, those reaching 1 kb, the manufactured subset of the latter, and those found by the 30x 2-of-3 consensus call set.
  An interval is manufactured when the longest benchmark record within it is itself shorter than 1 kb, so that it clears the floor only through fusion.
  (B) Percentage of benchmark intervals found by the same call set, with native and manufactured intervals scored separately.
  (C) Departures from one-to-one matching, at a permissive 0.1 classification threshold.
  The fixed 0.5 threshold was not used for this comparison because a query call cannot reach half of two benchmark intervals that do not themselves overlap.
  (D) Precision, recall, and F1 for the 30x 2-of-3 consensus at the 0.5 threshold.
  The dotted vertical line marks the 1 kb cap applied to padding in the joint parameter grid.
  Panels C and D are drawn for the 2-of-3 consensus; the same profiles for all six 30x call sets, together with the benchmark size distribution across the sweep, are given in Supplemental Figure Benchmark Padding Profiles.
]

#v(0.8em)

The truth set moves in two directions at once (Figure 4A).
Counted over all sizes the merged benchmark shrinks, from 178,838 intervals unpadded to 157,380 at 1 kb of padding and 79,071 at 100 kb, as records are absorbed into their neighbors.
Counted at or above the 1 kb floor it grows, from 24,820 intervals to 27,634 and then to 34,012 across the same range.
The second movement is the consequential one, since it is the intervals that clear the floor which form the recall denominator, and the growth is not a discovery of additional variants.
It is runs of sub-kilobase records fused into single intervals long enough to clear the 1 kb floor.

Those intervals can be counted directly.
An interval was labeled manufactured when the longest benchmark record inside it is itself shorter than 1 kb, so that the interval exists in the truth set only because padding joined the run.
Manufactured intervals are 337 of the 24,820 intervals in the unpadded truth set (1.4%), 3,911 of 27,634 at 1 kb of padding (14.2%), and 18,088 of 34,012 at 100 kb (53.2%).
They are also almost never recovered.
At 1 kb of padding the 30x 2-of-3 consensus finds 17.1% of the native intervals and 0.10% of the manufactured ones, a separation of more than two orders of magnitude that holds across the whole sweep (Figure 4B).
Padding therefore adds to the recall denominator a population that is by construction outside the resolution of every caller in this study.

The effect on the metrics follows from that arithmetic alone (Figure 4D).
Between zero padding and the 1 kb cap, the number of benchmark intervals the 2-of-3 consensus recovers rises from 4,042 to 4,062, an increase of 0.5%, while the denominator rises from 24,820 to 27,634, an increase of 11.3%.
Recall falls from 0.163 to 0.147 and F1 from 0.273 to 0.250 as a result, with no change in detection behind either.
Recall and F1 sit at their maxima at the bottom of the range for all six 30x call sets and decline monotonically above roughly 100 bp of padding (Supplemental Figure Benchmark Padding Profiles).

Precision behaves differently, and it is the one place in the sweep where padding does what it is intended to do.
It improves slightly over the first kilobase, reaching 0.841 at 1,543 bp of padding against 0.836 unpadded for the 2-of-3 consensus, because fusing benchmark fragments into single intervals converts a small number of boundary near-misses into matches.
The maxima for the other five call sets fall between 935 bp and 1,823 bp of padding, so the effect is real and consistently located, but it is worth less than one percentage point of precision.
Above a few kilobases precision falls with everything else, reaching 0.448 at 100 kb.

The matching itself is unaffected throughout.
At the 0.5 classification threshold used in this study, no query call matched more than one benchmark interval and no benchmark interval was split across more than one query call at any padding in the sweep, as the geometry of the threshold requires.
The matching is therefore strictly one-to-one over the entire range, and none of the movement above is an artifact of a single call being credited against several intervals at once.
Structure appears only in the permissive regime, and there it reverses direction (Figure 4C).
At a 0.1 threshold the number of query calls spanning more than one benchmark interval falls from 50 to 5 as fragments fuse into single partners, while the number of benchmark intervals split across more than one query call rises from 20 to a maximum of 38, both against totals of between 2,873 and 4,268 matched pairs.

Padding was therefore set to zero rather than capped.
There is no interior optimum available to select: recall and F1 are maximal at the bottom of the range, and the sub-percentage-point gain in precision available at a kilobase is bought with a 1.6 percentage point loss of recall.
Zero padding is not the same as disabling the operation, since it still bridges intervals that touch exactly; that difference amounts to 70 intervals out of 178,908, and it is retained because two benchmark records abutting at a shared coordinate describe one variant rather than two.
Padding is nonetheless carried through the joint parameter grid over \[0, 1 kb\], so that its interaction with the other three parameters is measured rather than assumed.

=== Size Floor

With padding fixed at zero, the merged benchmark holds 178,838 intervals across the thirteen samples, of which only 13.9% reach 1 kb and 2.8% reach 10 kb.
The preceding subsection took the 1 kb floor as given in order to isolate the effect of padding; this one asks whether that value is the right one.
The six 30x query call sets are between 7 and 75 times smaller than the benchmark and, because every caller was run at a 1 kb bin size, hold almost nothing below that width (Figure 5).
Scoring them against the benchmark without a size restriction therefore primarily measures the mismatch in resolution between the benchmark and the callers rather than an effect of sequencing depth, which is what this study set out to isolate.
We swept the floor symmetrically over both sides and examined how the comparison behaves as it rises (Figure 5).

#figure(
  image("/results/size_floor/detectable_size_domain_pub.png", width: 100%)
)

#cap("Figure 5:")[
  Effect of a size floor applied symmetrically to 30x WGS-derived query call sets and the merged benchmark call set, swept over \[1 bp, 100 kb\] at 80 logarithmically spaced points.
  (A) Number of intervals surviving the floor in each call set, with the merged benchmark as a dashed black line.
  (B) Recall, with the maximum attainable recall (query calls divided by truth intervals, capped at one) as a dashed line of the same color.
  (C) Precision.
  (D) F1, with each call set's maximum marked.
  The shaded band marks 1--5 kb, spanning every F1 maximum; the dotted vertical line marks the 1 kb floor adopted for all subsequent analyses.
  In panels C and D, each curve is drawn only while the call set behind it retains at least 100 intervals, since precision estimated from a few dozen calls is not comparable with precision estimated from thousands.
  The merged benchmark is 99.8% deletions, so these curves primarily describe deletion detection.
]

#v(0.8em)

Three features of this sweep together identify the usable domain.

First, the benchmark loses intervals far faster than any query call set.
Below roughly 500 bp the truth set outnumbers the largest query set by an order of magnitude, but it falls below the 1-of-3 consensus set above a 1,460 bp floor and below CNVpytor above 2,616 bp (Figure 5A).
Above those points the comparison has inverted: the callers report more CNVs than the benchmark contains, and precision is bounded by the size of the truth set rather than by the accuracy of the calls.

Second, recall is constrained by a ceiling that is a property of the two call set sizes rather than of detection (Figure 5B).
At an unrestricted floor (size floor = 0 bp) the maximum attainable recall is 0.135 for the 1-of-3 set and 0.013 for the 3-of-3 set, so even a caller that matched a distinct benchmark interval with every single one of its calls could not exceed those values.
Raising the floor lifts the ceiling for every call set, but it does so unevenly: the 1-of-3 and CNVpytor sets saturate at 1.0 above floors of 1,460 bp and 2,616 bp respectively, while Delly, GATK-gCNV, and the 2-of-3 and 3-of-3 consensus sets never approach it, reaching maxima of 0.55, 0.36, 0.31, and 0.18.
Recall is therefore interpretable as a detection measurement only below the point at which a given call set saturates.

Third, precision is flat from 1 bp to approximately 1 kb for every call set and declines above it (Figure 5C).
The flat region is the direct consequence of the bin size: over that range the floor removes benchmark intervals almost exclusively, because the callers had produced essentially nothing there to remove, so the query sets and their precision are unchanged while the truth set falls from 178,838 intervals to 24,820.
Above 1 kb the floor begins to remove query calls as well, and precision falls for every set, steeply for CNVpytor and the 1-of-3 consensus (0.318 to 0.052 and 0.317 to 0.051 between the unrestricted case and a 100 kb floor) and more gradually for GATK-gCNV, which is the most size-stable of the callers (0.566 to 0.448 over the same range, the upper end of which lies beyond the 64.6 kb floor at which its curve is cut for low counts).

Taken together, these place the usable domain immediately above the bin size.
We fixed the floor at 1 kb, chosen on the physical grounds that no caller in this study can resolve a CNV narrower than its bin.
The sweep supports that choice: F1 reaches its maximum between 1,689 bp and 4,051 bp for all six call sets, with the 2-of-3 consensus highest at 0.348 (Figure 5D).
A 1 kb floor therefore sits just below the empirical optimum for every call set simultaneously.

This floor is applied to every call set and to the benchmark for all analyses that follow, at all four coverages, and it is the value at which the floor is held while the two overlap thresholds are profiled below.
It was selected using 30x data only, which is the arm with the finest resolution and therefore the most permissive: a floor set there admits calls at 2x that fall below the resolution attainable at that depth, biasing the comparison against the low-coverage hypothesis rather than in its favor.

=== Consensus reciprocal overlap threshold
A consensus component is a union of all pairwise overlaps that clear the reciprocal overlap threshold.
We swept the threshold from 0.05 to 0.95 in steps of 0.05 (Figure 6).

#figure(
  image("/results/parameterization/consensus_overlap.png", width: 100%)
)

#cap("Figure 6:")[
  Effect of the query consensus reciprocal-overlap threshold on the three 30x consensus call sets, swept from 0.05 to 0.95.
  The benchmark is held at zero padding, the size floor at 1 kb on both sides, and the classification threshold at 0.5.
  (A) Consensus calls surviving the 1 kb floor.
  (B) Precision.
  (C) Recall, whose denominator is the same 24,820 benchmark intervals at every point.
  (D) F1.
  The dotted vertical line marks the adopted value of 0.5.
]

#v(0.8em)

The behavior for the number of calls greater than 1kb in length differs across consensus stringencies (Figure 6A).
Raising the threshold splits components between callers that fail to meet threshold requirements. Each split adds a call to lower stringency sets while removing the agreement that contributed to the counts in the levels above it.
Between 0.05 and 0.95 the 1-of-3 set grows from 21,847 to 28,413 calls above the floor, while the 2-of-3 set falls from 5,623 to 2,369 and the 3-of-3 set from 2,666 to 369.

Taking the ratio of a component's span to the span of the longest single call inside it, the median is 1.00 at every threshold and for every level, the 95th percentile never exceeds 1.16, and the largest ratio anywhere in the sweep is 2.83.
Components at the permissive end are therefore sets of calls that agree, not loci collapsed into intervals no caller reported.

What the threshold does instead is trade the two sides of the comparison against each other (Figure 6B, C).
For the 2-of-3 consensus, precision rises from 0.749 to 0.907 across the range while recall falls from 0.170 to 0.087.
For the 3-of-3 consensus precision is already near its ceiling and barely moves, from 0.912 to a maximum of 0.959 at 0.75, while recall falls from 0.098 to 0.014.
The 1-of-3 set is the exception: its recall is roughly flat 0.285 to 0.293 throughout; its precision falls to a minimum of 0.307 at 0.45 before rising to 0.392.

Over the lower half of the range those two movements cancel (Figure 6D).
Between 0.05 and 0.50 the F1 of the 2-of-3 consensus varies by 0.006, from 0.2785 at 0.30 to 0.2726 at 0.50, while its precision gains 8.7 percentage points.
The F1 maxima themselves are shallow and disagree between levels, falling at 0.30 for the 2-of-3 set, at 0.05 for the 3-of-3 set, and at the top of the range for the 1-of-3 set.

We chose to adopt a threshold of 0.5 for subsequent analyses.
It is the conventional reciprocal criterion, it is the smallest threshold at which neither member of a merged pair can be more than twice the size of the other, and it costs the 2-of-3 consensus 2.1% of its attainable F1 while gaining 8.7 of the 15.8 percentage points of precision available across the whole range.

=== Classification reciprocal overlap threshold

Unlike the other three parameters this one changes neither call set, and instead strictly influences binary classification calculations.
We swept the classification reciprocal overlap threshold, which defines the overlap required to register matches between query and truth sets as valid, from 0 to 0.99 in steps of 0.01 (Figure 7).

#figure(
  image("/results/parameterization/classification_overlap.png", width: 100%)
)

#cap("Figure 7:")[
  Effect of the classification reciprocal-overlap threshold, swept from 0 to 0.99.
  The benchmark is held at zero padding, the size floor at 1 kb on both sides, and the consensus threshold at 0.5.
  (A) Precision, recall, and F1 for the 30x 2-of-3 consensus.
  (B) Query calls credited against more than one benchmark interval, and (C) benchmark intervals credited to more than one query call, for all six 30x call sets; both axes are symmetric-log so that zero has a position.
  (D) F1 for the same six call sets.
  Panels B--D share the legend in D.
  The dotted vertical line marks the adopted value of 0.5.
]

#v(0.8em)

All three metrics decline monotonically and the parameter has no interior optimum (Figure 7A).
For the 2-of-3 consensus, precision falls from 0.876 at a threshold of zero to 0.836 at 0.5 and 0.410 at 0.9, and F1 from 0.291 to 0.273 to 0.134.
The choice is therefore not between values that perform differently but between definitions of what a match is required to mean, and the informative quantity is the match topology rather than the metrics.

At a threshold of zero a single shared base pair is a match, and one query call is credited against as many as 14 benchmark intervals, depending on the call set.
The number of query calls with more than one partner falls to zero between 0.41 and 0.47 depending on the call set, ahead of the 0.5 at which the geometry requires one-to-one matching (Figure 7B).
The merged benchmark is internally disjoint within each sample, chromosome, and variant type, so no query call can cover half of two of its intervals; the query sets carry no such guarantee, because components built at a reciprocal overlap of 0.5 may still overlap one another below that threshold.
The 1-of-3 consensus is the case in which that matters: it still splits 119 benchmark intervals across two query calls at a threshold of 0.5, and the count reaches zero only at 0.70 (Figure 7C).
For the other five call sets the matching is strictly one-to-one at 0.5, and the query-side and truth-side true-positive counts coincide exactly; for the 2-of-3 consensus they are both 4,042 at 0.5.

According to F1 scores, the 1-of-3 consensus is the highest-scoring set at every threshold up to 0.66 and the 2-of-3 consensus at every threshold above it, and the 3-of-3 set is the lowest until 0.90 (Figure 7D).
We adopted 0.5, the conventional reciprocal-overlap criterion, which is also the smallest threshold at which no query call can be credited against two benchmark intervals at once.

== Variance-based sensitivity analysis and Pareto front

The preceding subsections move one parameter with the other three held fixed.
To ask whether that reading survives elsewhere in the parameter field, all four were varied jointly over the full factorial grid of 22,743 settings and evaluated for each of the three consensus call sets at 30x.
The 2-of-3 consensus is reported here; the other two levels are given in Supplementary Table X.

#figure(
  image("/results/parameterization/sensitivity.png", width: 100%)
)

#cap("Figure 8:")[
  Joint behavior of the four parameters over the full factorial grid, for the 30x 2-of-3 consensus call set.
  (A) Sobol' first-order indices (solid) and total-order indices (pale extension) for precision, recall and F1, with the variance carried only by interactions at the right.
  (B) Marginal F1 of each parameter under the additive model, against position within that parameter's swept range; the dotted line is the grand mean.
  (C) Precision and recall at every setting in the grid (gray), the Pareto front (black), the adopted setting (star), and the setting attaining the highest F1 (diamond).
  (D) The same plane with the classification threshold held at 0.5, colored by size floor.
]

#v(0.8em)

The four parameters act very nearly independently (Figure 8A).
Their first-order indices sum to 0.94 for precision, 0.95 for recall and 0.95 for F1, so an additive model, a grand mean plus one curve per parameter, reproduces the joint field to within 5% of its variance.
The largest single interaction anywhere is between the size floor and the classification threshold, at 3.0% of the variance in F1 and 3.8% in precision.
The one-at-a-time profiles of the preceding subsections are therefore not artifacts of where the remaining parameters were held.

The variance is not shared evenly (Figure 8A, B).
The size floor carries 77.2% of the variance in F1 and 78.2% in recall, and the classification threshold carries 67.6% of the variance in precision.
The benchmark padding accounts for no more than 0.1% of the variance in any of the three metrics.
The same ordering holds at the 1-of-3 and 3-of-3 consensus levels, with one shift: the consensus threshold's share of the variance in F1 rises from 0.6% for the 1-of-3 set to 14.7% for the 3-of-3 set, which follows the intuition that for higher stringencies, changes in the threshold more readily remove calls.

These shares per parameter describe the region examined, and are influenced by the ranges we evaluate.
Recomputed over the deliberately over-wide grid, in which the padding and the floor extend to $10^6$ bp, the padding's share of the variance in F1 rises from 0.1% to 24.6% and the share carried by interactions rises from 4.7% to 25.9% (Supplementary Table X).
The padding's apparent inertness and the additivity of the field are properties of the ranges used, and are physically constrained around the values that we use for the subsequence analyses.

Across the grid, precision is between four and five times recall, and F1 is close to a monotone function of recall alone.
The Pareto front comprises 80 of the 22,743 settings, spanning recall from 0.033 to 0.241 and precision from 0.734 to 0.927 (Figure 8C).
Every setting on the front uses a classification threshold of 0.05, the loosest value tested in the grid.
The setting attaining the highest F1 of 0.363 uses zero padding and a 2 kb floor with consensus and classification thresholds of 0.05.
Both metrics decline monotonically in the classification threshold, so every value above the smallest is dominated; what this records is that a looser definition of a match credits more calls, not that the pipeline performs better under it.
The front is read with the crediting rule fixed for that reason.

Held at a classification threshold of 0.5, the front comprises 62 of the 1,197 remaining settings, and only two of them dominate the adopted setting: both raise the size floor to 2 kb, reaching a precision of 0.837 against the adopted 0.836 and a recall of 0.171 against 0.163 (Figure 8D).
Both therefore buy their advantage by scoring a smaller and larger-bodied set of events rather than by scoring the same events better.
With the floor also held at the 1 kb the caller resolution fixes and the front comprises 33 of the 171 settings that remain.
That front is traced almost entirely by the consensus threshold; the padding takes only three of its nine values along it, and moving from zero padding to 200 bp at the adopted consensus threshold gains 0.002 in precision for 0.002 in recall.
The choice of an operating point on this front is therefore a choice of consensus stringency, and the rate at which the two metrics exchange turns at the adopted value: raising the threshold from 0.05 to 0.50 gains 8.7 percentage points of precision for 0.7 of recall, while raising it from 0.50 to 0.95 gains a further 7.1 for 7.6.

== Adopted Parameters <r_adopted_parameters>

Each of the four parameters was bounded on geometric grounds and then measured across that bound, jointly as well as one at a time.
The values carried forward are the following, and all four are held at them for every call set and every coverage in the remainder of this study.

- *Benchmark padding, 0 bp.* The padding accounts for no more than 0.1% of the variance in any of the three metrics across its admissible range, so no value within that range is preferred on performance grounds. Zero is adopted because it is the only value at which the merged benchmark remains internally disjoint within a sample, chromosome, and variant type, and that disjointness is what the one-to-one crediting guarantee at a classification threshold of 0.5 rests on.
- *Size floor, 1 kb, applied symmetrically to the query and the benchmark.* No caller in this study can resolve a CNV narrower than its 1 kb bin. Below that width the comparison measures the resolution mismatch between the benchmark and the callers rather than an effect of sequencing depth, and above it the floor begins to remove query calls and precision falls for every call set. The value also sits immediately below the F1 maximum of all six 30x call sets, which fall between 1,689 bp and 4,051 bp.
- *Consensus reciprocal overlap, 0.5.* This is the conventional reciprocal criterion #c[Krusche 2019] #c[English 2022], and the smallest threshold at which neither member of a merged pair can exceed twice the size of the other. It costs the 2-of-3 consensus 2.1% of its attainable F1 and gains 8.7 of the 15.8 percentage points of precision available across the range.
- *Classification reciprocal overlap, 0.5.* This is the same criterion applied to the second comparison, and is the smallest threshold at which no query call can be credited against two benchmark intervals at once, so that a true positive count is a count of distinct events on both sides.

The two overlap thresholds are numerically equal but independent: the first decides which calls from different callers become one consensus call, the second decides which consensus calls are credited against the benchmark.
Under these values the merged benchmark holds 24,820 intervals above the floor and the 30x 2-of-3 consensus call set holds 4,836 calls.
Recall is reported against the whole merged benchmark throughout, so it carries a ceiling equal to the ratio of the two set sizes, which is 0.195 for the 30x 2-of-3 set; that ceiling is stated wherever call sets of different sizes are compared.

With the comparison fixed, we return to the question this study set out to answer.
Low-pass WGS has been shown to support cost-effective CNV calling in several settings #c[Kucharík 2021] #c[Mazzonetto 2024] #c[Mazzonetto 2024 (2)], but not at the coverages a BGE run delivers alongside its exome data.
The sections that follow describe the call sets themselves, establish which consensus level to carry forward, and then compare that level across coverages.

== CNV Size Distribution Characteristics <r_size_distributions>

Size distributions of CNVs in the four call-set constructions in this study differ significantly, restricting which events a given method can reasonably identify.
Every set below is taken at the adopted parameters, so consensus components are built at a reciprocal overlap of 0.5 and the 1 kb floor is applied to every set including the benchmark and the array.
No upper bound is imposed.

#v(0.5em)

#block(width: 100%)[
#set text(hyphenate: false, size: 10pt)
#table(
  columns: (1fr, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right),
  table.header(
    [Call Set], [Total Call Count], [Min. Size], [Median Size], [MAD], [Max. Size], [\% DUP],
  ),
  [*30x -- CNVpytor*], [16,159], [2,000], [14,000], [11,000], [715,000], [23.5],
  [*30x -- Delly*], [6,776], [1,004], [4,327], [2,624], [4,019,216], [27.5],
  [*30x -- GATK-gCNV*], [8,216], [1,000], [3,000], [2,000], [516,000], [10.9],
  table.hline(stroke: 0.3pt),
  [*30x -- 1/3 Consensus*], [23,876], [1,000], [6,000], [4,837], [4,019,216], [25.0],
  [*30x -- 2/3 Consensus*], [4,836], [1,057], [5,343], [2,343], [925,483], [9.8],
  [*30x -- 3/3 Consensus*], [2,400], [2,000], [6,186], [2,186], [703,537], [4.8],
  table.hline(stroke: 0.3pt),
  [*SNP Array*], [1,548], [1,003], [8,567], [5,969], [776,884], [45.9],
  [*Merged Benchmark*], [24,820], [1,000], [3,176], [1,885], [1,236,070], [1.4],
)
]

#cap("Table 8:")[
  CNV size statistics of the 30x call sets, the SNP microarray call set, and the merged benchmark, in base pairs.
  Spread is reported as the median absolute deviation, which is the median of the absolute deviations from the median, rather than as a standard deviation: sizes span three orders of magnitude with a heavy right tail, over which a moment-based spread is set by a handful of the largest calls.
  Median and MAD are rounded to the nearest base pair.
  Every call in every set is a deletion or a duplication, so the deletion share is the complement of the column given.
  The same statistics for all four coverages, together with the mean and the quartiles, are given in Supplemental Table Size Distribution Statistics.
]

#v(0.8em)

#figure(
  image("/results/size_distribution/size_distributions.png", width: 100%)
)

#cap("Figure 9:")[
  Size densities of the CNV call sets, estimated by Gaussian kernel density on $log_10$ size and drawn on a log-10 axis.
  (A) The six 30x call sets, the SNP microarray call set, and the merged benchmark.
  (B--D) Each consensus level at all four coverages, with the merged benchmark repeated as a dashed reference; the three panels share a vertical scale.
  All sets are restricted to the adopted 1 kb floor, which truncates each density on the left.
  The equivalent panels for the individual callers are given in Supplemental Figure Per Caller Size Distributions.
]

#v(0.8em)

The three callers do not resolve breakpoints in the same way.
CNVpytor reports no call shorter than 2 kb and every call it makes is an exact multiple of the 1 kb bin, consistent with a read-depth segmentation that requires two adjacent bins to agree #c[Suvakov 2021].
GATK-gCNV is bin-quantized in the same way but reaches a single bin, so 24.9% of its calls fall between 1 and 2 kb #c[Babadi 2023].
Delly is the exception: only 0.03% of its calls land on a bin boundary and its shortest is 160 bp, since its breakpoints come from split reads and read pairs rather than strictly from a depth profile #c[Rausch 2012].
The 1 kb floor therefore removes nothing from CNVpytor or GATK-gCNV and 3.5% of Delly's calls.

Consensus construction narrows the distribution (Figure 9A).
At 30x the three consensus levels have medians of 6,000, 5,343 and 6,186 bp , and the MAD falls from 4,837 to 2,343 and 2,186 bp as stringency increases (Table 8).
Calls between 1 and 2 kb make up 12.8% of the 1-of-3 set, 2.6% of the 2-of-3 set and none of the 3-of-3 set.
Since calls of that width have to be matched at a reciprocal overlap of 0.5 to survive, and CNVpytor, which is in every 3-of-3 component, reports nothing below 2 kb, the minimum reported size is 2kb for the 3-of-3 set.
Callers agree on a duplication considerably less often than on a deletion, with the percentage of duplications calls falling from 25.0% of the 1-of-3 set to 9.8% and 4.8% of the 2-of-3 and 3-of-3 sets (Table 8).

The two reference sets differ significantly from the sequence-derived call sets.
Only 13.9% of the 178,838 merged benchmark intervals reach 1 kb, and the 24,820 that do carry the tightest distribution in the study, with 31.3% of their mass between 1 and 2 kb.
The array is the opposite: 93.3% of its 1,659 calls clear the floor, and those have the largest median and the widest spread of any set.
The two differ in class more sharply than in size, at 45.9% duplications in the SNP Array against 1.4% in the merged benchmark (Table 8).

Decreasing coverage moves the distributions to the right.
Delly's median slightly rises from 4,327 bp at 30x to 5,316 bp at 2x, while CNVpytor's rises from 14,000 to 51,000 bp (Supplemental Figure Per Caller Size Distributions).
Consensus calls are built out of the callers' own intervals, so the consensus sets widen with them (Figure 9B--D): the 2-of-3 median rises from 5,343 bp at 30x to 11,771, 21,000 and 33,000 bp at 6x, 4x and 2x, and the fraction of that set above 10 kb rises from 23.1% to 85.0%.
Counts fall across the same range for the 2-of-3 and 3-of-3 sets, from 4,836 to 721 and from 2,400 to 177, but not for every set: GATK-gCNV reports 8,216 calls at 30x and 10,553 at 2x, carrying the 1-of-3 consensus back up from 14,335 at 4x to 18,907 at 2x.

The class composition drifts with depth as well.
Duplications are 9.8% of the 30x 2-of-3 set and 47.0% of the 2x set, a difference in survival rather than in discovery: deletions in that set fall from 4,360 to 382 across the range while duplications fall only from 476 to 339.
The merged benchmark is 1.4% duplications, so the low-coverage sets move away from its composition as they lose calls, which bears on every recall measurement made at those depths.

The array's size profile falls between the 30x and 6x consensus sets rather than beside either, with a median of 8,567 bp against 5,343 and 11,771 bp and 45.5% of its calls above 10 kb against 23.1% and 56.3%.
Array and low-pass sequencing are therefore confident over similar size ranges despite drawing on different evidence, which is the comparison the rest of the Results takes up.

== Consensus Level Selection <r_consensus_levels>

The six 30x call sets and the SNP array were scored against the merged benchmark at the four adopted parameters (Table 9).
Precision rises with the number of callers required, from 0.307 for the 1-of-3 set to 0.836 for the 2-of-3 set and 0.948 for the 3-of-3 set, and both agreement levels exceed every individual caller.
Recall carries a ceiling equal to the ratio of the call set size to the 24,820 benchmark intervals, which runs from 0.962 for the 1-of-3 set to 0.097 for the 3-of-3 set.

#v(0.5em)

#block(width: 100%)[
#set text(hyphenate: false, size: 10pt)
#table(
  columns: (1fr, auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right, right, right),
  table.header(
    [Call Set], [Calls], [TP], [FP], [Precision], [Precision (DEL)], [Recall], [Max. Recall], [F1],
  ),
  [*30x -- CNVpytor*], [16,159], [5,131], [11,028], [0.318], [0.407], [0.207], [0.651], [0.250],
  [*30x -- Delly*], [6,776], [3,941], [2,835], [0.582], [0.786], [0.159], [0.273], [0.250],
  [*30x -- GATK-gCNV*], [8,216], [4,542], [3,674], [0.553], [0.611], [0.183], [0.331], [0.275],
  table.hline(stroke: 0.3pt),
  [*30x -- 1/3 Consensus*], [23,876], [7,340], [16,536], [0.307], [0.403], [0.291], [0.962], [0.299],
  [*30x -- 2/3 Consensus*], [4,836], [4,042], [794], [0.836], [0.912], [0.163], [0.195], [0.273],
  [*30x -- 3/3 Consensus*], [2,400], [2,276], [124], [0.948], [0.978], [0.092], [0.097], [0.167],
  table.hline(stroke: 0.3pt),
  [*SNP Array*], [1,548], [514], [1,034], [0.332], [0.579], [0.021], [0.062], [0.039],
)
]

#cap("Table 9:")[
  Binary classification of the 30x call sets and the SNP array against the merged benchmark, at the adopted parameters.
  A true positive is a call that clears the classification threshold against at least one of the 24,820 benchmark intervals above the size floor, and a false positive is a call that clears it against none.
  Recall is the fraction of those 24,820 intervals matched by at least one call, and Max. Recall is the largest value recall could take for a call set of that size, being the number of calls divided by the number of benchmark intervals, capped at one.
  The matching is one-to-one for every set but the 1/3 consensus, so the number of matched calls equals the number of matched benchmark intervals; the 1/3 set's 7,340 matched calls cover 7,221 intervals.
  A call is never matched against a benchmark interval of the other variant class, so deletions and duplications partition every count in the table exactly.
  Precision (DEL) is therefore the precision of the deletion half of the same classification; the duplication half is given in Supplemental Table Performance by Variant Class.
]

#v(0.8em)

F1 is highest for the 1-of-3 set, at 0.299 against 0.273 for the 2-of-3 set, but that ordering follows from call set size rather than from detection.
The 1-of-3 set reaches its higher recall by holding 4.9 times as many calls as the 2-of-3 set, 16,536 of which match nothing in the benchmark for the former against 794 in the latter.

Splitting the 1-of-3 set by the number of callers that reported each component locates those calls (Figure 10B).
The 19,040 components carried by a single caller have a precision of 0.173, the 2,436 carried by exactly two have 0.725, and the 2,400 carried by all three have 0.948.
The single-caller set holds 15,742 of the 16,536 unmatched calls, and it is this set that the single-population agreement model could not account for, so its members are better described as caller-specific artifacts than as events beyond the reach of the other two callers.

Requiring a second caller removes 95.2% of the false positives and 44.9% of the true positives, and requiring a third removes a further 670 false positives and 1,766 true positives, buying 0.112 of precision for 43.7% of the matches that remained.
Every call in a set destined for functional validation costs about the same to follow up, so requiring two callers leaves a smaller candidate set with a higher chance of holding something functionally relevant; requiring a third buys a marginal gain in confidence with true positives that may be worth more, depending on the application #c[Ho 2020] #c[Liu 2022].

The benchmark holds 24,477 deletions and 343 duplications above the floor, so a duplication metric rests on a truth set two orders of magnitude thinner than the deletion one and is not comparable with it.
Restricted to deletions the ordering is unchanged and the separation between levels is wider, at precisions of 0.403, 0.912 and 0.978 (Table 9).

#figure(
  image("/results/consensus_levels/consensus_levels.png", width: 100%)
)

#cap("Figure 10:")[
  Performance of the 30x call sets and the SNP array against the merged benchmark at the adopted parameters.
  (A) Precision against recall, over gray F1 iso-contours.
  Each point sits on a bar running to its recall ceiling, the largest recall a call set of that size could attain.
  (B) Precision of the 1/3 consensus call set split by the number of callers that reported each component, taken over exactly that many callers rather than at least that many.
  (C) F1 against CNV size, smoothed in log-10 space over a bandwidth of 0.15 decades and drawn only where at least 50 calls contribute.
  Panels A and C share the legend in A.
]

#v(0.8em)

F1 peaks between 4.0 and 7.3 kb for every call set (Figure 10C).
The 2-of-3 set has the highest peak of the six, 0.424 near 5.5 kb, and is above the 1-of-3 set at every size over 4,248 bp, so its aggregate F1 disadvantage is confined to the narrowest calls the comparison admits.

The SNP array falls below every sequence-derived call set on all three metrics, at a precision of 0.332, a recall of 0.021 and an F1 of 0.039.
Composition accounts for much of that: 45.9% of its calls are duplications (Table 8), the class the benchmark barely represents, and its deletion precision of 0.579 sits within the range of the individual callers.

We carried the 2-of-3 consensus into the coverage comparison.
Its precision of 0.836 leaves a candidate set clean enough to act on while recovering 4,042 benchmark intervals, against 7,221 intervals at a precision of 0.307 for the 1-of-3 set and 2,276 at 0.948 for the 3-of-3 set.
The 3-of-3 set is contained within it, so 2,400 of those 4,836 calls carry the support of all three callers and thus admit the subset of calls with the highest confidence for functional relevance.

== Performance of 2-of-3 Consensus Call Sets across Coverages <r_coverage_performance>

The 2-of-3 consensus call sets from all four coverages and the SNP array were scored against the merged benchmark at the adopted parameters (Table 10).
Precision falls from 0.836 at 30x to 0.746, 0.671 and 0.502 at 6x, 4x and 2x, and recall falls from 0.163 to 0.015 against a recall ceiling that falls from 0.195 to 0.029.

#v(0.5em)

#block(width: 100%)[
#set text(hyphenate: false, size: 10pt)
#table(
  columns: (1fr, auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right, right, right),
  table.header(
    [Call Set], [Calls], [TP], [FP], [Precision], [Precision (DEL)], [Recall], [Max. Recall], [F1],
  ),
  [*30x -- 2/3 Consensus*], [4,836], [4,042], [794], [0.836], [0.912], [0.163], [0.195], [0.273],
  [*6x -- 2/3 Consensus*], [1,708], [1,274], [434], [0.746], [0.909], [0.051], [0.069], [0.096],
  [*4x -- 2/3 Consensus*], [1,119], [751], [368], [0.671], [0.943], [0.030], [0.045], [0.058],
  [*2x -- 2/3 Consensus*], [721], [362], [359], [0.502], [0.893], [0.015], [0.029], [0.028],
  table.hline(stroke: 0.3pt),
  [*SNP Array*], [1,548], [514], [1,034], [0.332], [0.579], [0.021], [0.062], [0.039],
)
]

#cap("Table 10:")[
  Binary classification of the 2/3 consensus call sets across coverages and the SNP array against the merged benchmark, at the adopted parameters.
  A true positive is a call that clears the classification threshold against at least one of the 24,820 benchmark intervals above the size floor, and a false positive is a call that clears it against none.
  Recall is the fraction of those 24,820 intervals matched by at least one call, and Max. Recall is the largest value recall could take for a call set of that size, being the number of calls divided by the number of benchmark intervals, capped at one.
  The matching is one-to-one for every set listed, so the number of matched calls equals the number of matched benchmark intervals.
  A call is never matched against a benchmark interval of the other variant class, so deletions and duplications partition every count in the table exactly.
  Precision (DEL) is therefore the precision of the deletion half of the same classification; the duplication half is given in Supplemental Table Performance by Variant Class.
]

#v(0.8em)

Deletion precision does not fall with depth, holding at 0.912, 0.909, 0.943 and 0.893 across the four coverages (Table 10), while duplication precision falls from 0.137 to 0.062 (Supplemental Table Performance by Variant Class).
Reweighting each set's class-wise precisions to the 30x class composition gives 0.836, 0.831, 0.860 and 0.811, so the fall in overall precision is the drift towards duplications rather than a loss of accuracy within either class.

#figure(
  image("/results/coverage_performance/benchmark_recovery.png", width: 100%)
)

#cap("Figure 11:")[
  Recovery of the merged benchmark by the 2/3 consensus call sets across coverages and by the SNP array, at the adopted parameters.
  (A) UpSet plot of the benchmark intervals each call set recovered.
  Each bar counts the intervals recovered by exactly the combination of call sets marked beneath it, and the horizontal bars at left give each call set's total.
  The twelve largest of the 27 non-empty combinations are drawn, holding 4,088 of the 4,138 recovered intervals; the 20,682 intervals no call set recovered are not shown.
  (B) The two containment fractions relating each coverage's recoveries to the array's: the fraction of the intervals recovered by the array that the consensus set also recovered, and the fraction of those recovered by the consensus set that the array also recovered.
]

#v(0.8em)

All five call sets were classified against the same 24,820 benchmark intervals, so the intervals each recovered can be crossed directly (Figure 11A).
4,138 intervals, 16.7% of the benchmark, are recovered by at least one of the five.
The coverage sets are close to nested: 96.5% of the intervals recovered at 6x are also recovered at 30x, 91.7% of those recovered at 4x are recovered at 6x, and 92.5% of those recovered at 2x are recovered at 4x.
The four largest combinations follow this nesting, with 2,560 intervals recovered by the 30x set alone and 463, 284 and 208 as each successive coverage is added.

The 30x set recovers 471 of the 514 benchmark intervals the array recovers, or 91.6%, along with 3,571 the array does not, leaving 43 intervals to the array alone.
That share falls to 50.8%, 40.1% and 23.5% at 6x, 4x and 2x, while the share of the sequencing set's own recoveries that the array also holds rises from 11.7% to 33.4% (Figure 11B).
The two cross between 4x and 2x: the 2x set recovers 241 intervals the array does not against 393 recovered by the array alone, and 362 in total against the array's 514.

#figure(
  image("/results/coverage_performance/coverage_size_metrics.png", width: 100%)
)

#cap("Figure 12:")[
  Precision (A), recall (B) and F1 (C) against CNV size for the 2/3 consensus call sets across coverages and the SNP array, at the adopted parameters.
  All three are smoothed in log-10 space over a bandwidth of 0.15 decades and drawn only where at least 20 calls contribute, which is why the lower-coverage curves do not extend to the size floor.
  All three panels share the legend in A.
]

#v(0.8em)

Precision above 10 kb is nearly independent of depth, peaking at 0.881, 0.910 and 0.888 for 30x, 6x and 4x and 0.758 for 2x (Figure 12A).
Recall separates the coverages instead, and the separation narrows as CNVs get larger: at 5 kb it runs 0.279 at 30x against 0.008 at 2x, and at 50 kb 0.123 against 0.077 (Figure 12B).
The F1 peak moves right and down accordingly, from 0.424 near 5.5 kb at 30x to 0.133 near 45.2 kb (Figure 12C).
The 4x and 2x sets hold too few calls below 2.1 and 2.9 kb for any of the three metrics to be estimated there.

The array's F1 peaks at 0.100 near 12.5 kb, below the peak of every coverage including 2x, and its precision above 10 kb is 0.435 against 0.758 for the weakest of the sequencing sets.
Lowering coverage removes small CNVs from the call set rather than degrading the calls that survive: the 2x set's deletion precision is within 0.02 of the 30x set's, and its recall deficit closes from 35-fold at 5 kb to 1.6-fold at 50 kb.

= Discussion

With sequencing becoming increasingly ubiquitous in both research and clinical laboratories, a question has emerged for CNV analysis: can the low-pass, genome-wide data from the BGE workflow substitute SNP genotyping arrays as an approach for CNV detection in the size regimes typically targeted by microarray testing?
BGE was developed to pair deep exome variant discovery with economical, low-pass genome-wide coverage in a single sequencing product, offering a potential path to consolidate CNV detection into existing sequencing pipelines rather than maintaining parallel array workflows and cross-platform data harmonization #c[DeFelice 2024] #c[Boltz 2026].
In this study, we directly evaluated the "array-replacement" proposition by benchmarking CNV calls from short-read lcWGS against SNP-array derived CNV calls, while using high-coverage WGS as a complementary reference point to clarify which performance differences are primarily coverage-limited versus method-limited.

Aligning with discoveries made by other studies #c[Li and Olivier 2012], our results demonstrated that high-coverage sequencing-based approaches result in recovery of considerably more CNVs compared to SNP array-based alternatives, with particularly higher call rates in the several kilobase regime.
Coverages of 2x -- 6x that are typically retrieved in current-day BGE pipelines seem to perform best in a mid-to-large CNV regime, similar to typical results of SNP arrays.
Call sets at 4x coverage and above showed consistently higher precision and recall than SNP arrays across all sizes tested, with the performance scales substantially with the coverage.
At 2x — the lowest coverage evaluated — overall performance was broadly comparable to the SNP array, with the 2x call set achieving substantially higher precision (0.564 vs. 0.336) while the SNP array modestly exceeded it in recall (0.062 vs. 0.048) and marginally in F₁ and F₂ scores.
The results illustrate the high performance scalability of low-coverage sequencing-based CNV calling: recall nearly doubles when moving from 2x to 4x — surpassing SNP array performance across most of the analysis window — with a similar magnitude of increase demonstrated going from 4x to 6x.

BGE incurs roughly 28% of the per-sample cost of deep WGS (\~\$99 vs. \$350) while remaining cost-comparable to a GWAS array #c[Boltz 2026].
Combined with its ease of adaptability and the scalability of sequencing, this narrows the competitive margin of SNP array–based calling over sequencing-based approaches.
This is particularly promising compared to the relatively limited scalability of SNP array technology: instead of simply rerunning genomic samples through sequencing pipelines with different parameters, expanding array detection requires the purchasing of new microarray chips and associated infrastructure, as well as potentially modifying existing computational workflows to handle the new data.
As a result, lcWGS-based CNV detection has already seen some adaptation as a promising alternative to microarray-based analyses #c[Boltz 2026] #c[Wang 2019].

Several other tools were either dismissed as inappropriate for use with the data analyzed in this study or introduced significant complexity for negligible differences in performance compared to the three tools we had chosen (Supplemental Table Tools).
Other studies #c[Gabrielaite 2021] have made it clear that the use of tools designed specifically for WES data are significantly outperformed by tools that use WGS data.
ZipCNV #c[Xue 2025], a tool that uses base level read-depth information along with dynamic sliding windows to smooth depth signals and more accurately identify CNVs, was tested as a promising candidate tool.
However, our attempts at using the tool on the high-performance cluster used for this study were unsuccessful due to the program demonstrating major I/O and memory usage issues.
LUMPY #c[Layer 2014] was considered because of its strong reported performance and its probabilistic framework for integrating multiple SV signals.
In LUMPY, read-pair, split-read, and read-depth evidence are first modeled separately as breakpoint probability distributions and then clustered into a joint breakpoint prediction.
However, because our consensus set already included two read-depth-based callers (CNVpytor and GATK-gCNV), we prioritized inclusion of a caller whose evidence integration was less likely to preserve read-depth-specific artifacts in parallel.
DELLY instead uses discordant paired-end clusters to nominate breakpoint-containing intervals and then refines these candidates using split-read support to achieve higher-resolution breakpoint definition.
We therefore selected DELLY over LUMPY to increase methodological complementarity across callers and reduce the likelihood that technology- or algorithm-specific artifacts would propagate into the consensus call set.

Duplications are a known hard class to detect in short-read CNV/SV analysis and often show more caller disagreement and metric sensitivity #c[Ho 2020].
Independent lcWGS benchmarking echoes this: amplification calls diverged most across callers and were prone to over-detection in sparse data, whereas deletion calls remained comparatively stable #c[Wang 2025].
This is consistent across all of our obtained results, which demonstrated that duplication-only CNV calls yielded much lower performance, much higher variance, and different performance distribution trends compared to the deletion-only calls.
This expected behavior was the primary motivation behind stratifying all results across the two primary types of structural variation.

// NOTE(lionel): condensed on 2026-08-17 from the original intersection-vs-union
// paragraph, since that axis was cut from Methods and Results. The [English 2022]
// Truvari citation is retained here. Revisit on your Discussion pass.
Consensus components were collapsed by union, so a merged call spans the minimum start and the maximum end position across its member calls.
Work from other SV analysis toolkits such as Truvari has demonstrated that slight differences in the implementation details of merging and matching SV calls propagate into substantial differences in results #c[English 2022], so this policy warrants being stated explicitly rather than treated as an incidental detail.
Union merging may aggregate adjacent but distinct CNVs where callers disagree on breakpoint position, or where one caller preferentially reports larger intervals than the others.
It also retains whichever member call carries the most permissive breakpoints, which in our data was frequently Delly: CNVs called by Delly that had equivalent calls from another tool often had inferred breakpoints extending past the 1 kb bin boundaries used by CNVpytor and GATK-gCNV (Supplemental Figure Modulus Size Distribution).
Union merging therefore preserves more of Delly's breakpoint-specific information than a policy restricted to the region of agreement between callers would.

A notable limitation of this study is the absence of a verifiably comprehensive benchmark set.
The merged benchmark used here aggregates structural variant calls from three datasets #c[1000G 2015]#c[Logsdon 2025]#c[Schloissnig 2024], — each produced by different calling algorithms and parameterization strategies.
As a result, a substantial proportion of benchmark CNVs were expected to be undetectable by the short-read, depth-based methods evaluated in this study, either due to size constraints imposed by 1kb binning, coverage limitations, or fundamental differences in detection methodology between the benchmark sources and our evaluated call sets.
To keep recall interpretable, false negatives were therefore restricted to CNVs discovered by at least one call set in this study; As expected, the set of discoverable CNVs represented only a fraction of the total benchmark set (\~9.73%).
Furthermore, the 50% reciprocal overlap threshold used for matching may have rejected genuine CNV pairs due to breakpoint disagreement between datasets, and some calls flagged as false positives may correspond to real variants absent from the benchmark rather than true errors.
Despite these limitations, the relative performance differences between the evaluated call sets remain consistent and interpretable across all metrics reported.
Finally, the tested sample size (n = 13) was relatively small — constrained by the limited overlap between samples with both benchmark datasets and SNP microarray data.

Beyond discoverability, the benchmark required reference-build harmonization: the 1000G phase 3 call set and the SNP microarray data were lifted over to GRCh38/hg38 (from GRCh37/hg19 and NCBI36/hg18, respectively), whereas the remaining sources were natively aligned to GRCh38.
Liftover between human assemblies reconciles the large majority of loci, and only X% of CNVs were lost during conversion in our pipeline; nonetheless, the microarray is the more liftover-susceptible of the two sources, since aggregation with the natively-aligned benchmark sets partially buffers the merged benchmark against coordinate loss.

Collectively, our findings support lcWGS-based CNV calling from BGE pipelines as a pragmatic pathway to replace the ongoing complexity of storing, curating, and harmonizing microarray outputs with sequencing outputs—while still capturing a significant number of CNVs to be viable for cytogenetic detection.
This proposed approach is especially practical when sequencing data (or sequencing infrastructure) is already available; conversely, SNP-array testing may remain advantageous for legacy laboratory pathways or settings where computational capacity is the overriding constraint.
Building on this work, our next step is to validate performance on real BGE sequencing data — rather than downsampled high-coverage WGS — and to characterize performance at coverage depths that may reflect higher-yield BGE configurations #c[Boltz 2026].
An additional future direction to explore includes length-only stratification by benchmarking separately across genomic contexts known to challenge CNV discovery—segmental duplications, tandem repeats/low-complexity sequence, and low-mappability or GC-extreme regions—using established stratification resources and region definitions #c[Krusche 2019] #c[Dwarshuis 2024].
Together with expanded sample size and targeted validation of calls, these additions should allow us to state more precisely when lcWGS can fully supplant microarrays for CNV detection and where traditional array-based approaches remain warranted.

In conclusion, to our knowledge we have performed the first performance comparison between high-coverage, low-coverage, and SNP-array based CNV detection using modern CNV calling tools.
Our work provides a benchmark evaluation methodology extensible to low-coverage sequencing data using multiple benchmark CNSV collections, rooted in the paradigms of prior studies #c[Masood 2024] #c[Wang 2025].
We have created a solid foundation for the aggregation of several popular, mathematically rigorous, and well adapted CNV calling tools to optimize the recovery of true candidate CNVs.

= Supplementals

#link("https://docs.google.com/document/d/1570MKXc9A6cBSIJ-BAegH1mik7SeUY4wc4HnASq4neA/edit?usp=sharing")[CNV Benchmark Paper - Supplementals]

#link("https://docs.google.com/spreadsheets/d/1VAIRYjwxRzuHfRzZ0Ap5H1hdbB0husPFGBHiKqbmgHE/edit?usp=sharing")[CNV Benchmark Paper - Supplementals Spreadsheet]

= References

#set par(justify: false)
#set text(size: 10pt)

#link("https://www.nature.com/articles/nrg3871") #c[Zarrei 2015] \
#link("https://www.sciencedirect.com/science/article/pii/S0167527319301913") #c[Glessner 2020] \
#link("https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2013.00092/full") #c[Valsesia 2013] \
#link("https://www.sciencedirect.com/science/article/pii/S0888754324001836?via%3Dihub#s0005") #c[Baardwijk 2024] \
#link("https://journals.physiology.org/doi/full/10.1152/physiolgenomics.00082.2012#sec-5") #c[Li and Olivier 2012] \
#link("https://www.nature.com/articles/s41431-022-01162-2") #c[Ewans 2022] \
#link("https://www.biorxiv.org/content/10.1101/2024.04.03.587209v3.abstract") #c[DeFelice 2024] \
#link("https://www.nature.com/articles/s41588-026-02669-w") #c[Boltz 2026] \
#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC11188507/") #c[Masood 2024] \
#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC12481693/") #c[Wang 2025] \
#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC7402362/") #c[Ho 2020] \
#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC7042067/") #c[Wang 2019] \
#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC8071346/") #c[Kucharík 2021] \
#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC9793516/") #c[English 2022] \
#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC6699627/") #c[Krusche 2019] \
#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC11489684/") #c[Dwarshuis 2024] \
#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC8454654/") #c[Zook 2020] \
#link("https://www.biorxiv.org/content/10.1101/2021.02.06.430068v1.full") #c[Byrska-Bishop 2021]

#v(0.6em)

*Tool Comparison:* #link("https://pmc.ncbi.nlm.nih.gov/articles/PMC8699073/") #c[Gabrielaite 2021] \
*ZIPcnv:* #link("https://academic.oup.com/bioinformatics/article/41/11/btaf592/8301287") #c[Xue 2025] \
*LUMPY:* #link("https://pmc.ncbi.nlm.nih.gov/articles/PMC4197822/") #c[Layer 2014]

#v(0.6em)

*PennCNV:* Wang K, Li M, Hadley D, Liu R, Glessner J, Grant SF, Hakonarson H, Bucan M.
PennCNV: an integrated hidden Markov model designed for high-resolution copy number variation detection in whole-genome SNP genotyping data.
Genome research. 2007 Nov 1;17(11):1665-74. #c[Wang 2007]

#v(0.6em)

*Consensus Calling Justification:* #link("https://link.springer.com/article/10.1186/s13059-022-02636-8") #c[Liu 2022]

#v(0.6em)

#link("https://pubmed.ncbi.nlm.nih.gov/23329113/") #c[Weischenfeldt 2013] \
#link("https://pubmed.ncbi.nlm.nih.gov/29396143/") #c[Hu 2018] \
#link("https://pubmed.ncbi.nlm.nih.gov/39643102/") #c[Kushima 2025] \
#link("https://pubmed.ncbi.nlm.nih.gov/20164920/") #c[Beroukhim 2010] \
#link("https://pubmed.ncbi.nlm.nih.gov/19566914/") #c[Shlien 2009] \
#link("https://pubmed.ncbi.nlm.nih.gov/38565148/") #c[Lemire 2024] \
#link("https://pubmed.ncbi.nlm.nih.gov/28963714/") #c[Nowakowska 2017]

#v(0.6em)

#link("https://pubmed.ncbi.nlm.nih.gov/33920867/") #c[Kucharík 2021] \
#link("https://pubmed.ncbi.nlm.nih.gov/37807935/") #c[Mazzonetto 2024] \
#link("https://pubmed.ncbi.nlm.nih.gov/38924610/") #c[Mazzonetto 2024 (2)]

#v(0.6em)

*cyvcf2:* Pedersen BS, Quinlan AR. cyvcf2: fast, flexible variant analysis with Python.
Bioinformatics. 2017 Jun 15;33(12):1867-9.
#link("https://academic.oup.com/bioinformatics/article/33/12/1867/2971439") #c[cyvcf2]

#v(0.6em)

*IGSR:* Fairley S, Lowy-Gallego E, Perry E, Flicek P. The International Genome Sample Resource (IGSR) collection of open human genomic variation resources.
Nucleic acids research. 2020 Jan 8;48(D1):D941-7. #c[Fairley 2020] \
*1000G:* 1000 Genomes Project Consortium. A global reference for human genetic variation.
Nature. 2015 Sep 30;526(7571):68. #c[1000G 2015] \
*HGSVC3:* Logsdon GA, Ebert P, Audano PA, Loftus M, Porubsky D, Ebler J, Yilmaz F, Hallast P, Prodanov T, Yoo D, Paisie CA.
Complex genetic variation in nearly complete human genomes. Nature. 2025 Aug 14;644(8076):430-41. #c[Logsdon 2025] \
*ONT Vienna:* Schloissnig S, Pani S, Rodriguez-Martin B, Ebler J, Hain C, Tsapalou V, Söylev A, Hüther P, Ashraf H, Prodanov T, Asparuhova M.
Long-read sequencing and structural variant characterization in 1,019 samples from the 1000 Genomes Project.
bioRxiv. 2024 Apr 20:2024-04. #c[Schloissnig 2024]

#v(0.6em)

*CNVpytor:* Suvakov M, Panda A, Diesh C, Holmes I, Abyzov A. CNVpytor: a tool for copy number variation detection and analysis from read depth and allele imbalance in whole-genome sequencing.
Gigascience. 2021 Nov;10(11):giab074. #c[Suvakov 2021] \
*GATK:* McKenna A, Hanna M, Banks E, Sivachenko A, Cibulskis K, Kernytsky A, Garimella K, Altshuler D, Gabriel S, Daly M, DePristo MA.
The Genome Analysis Toolkit: a MapReduce framework for analyzing next-generation DNA sequencing data.
Genome research. 2010 Sep 1;20(9):1297-303. #c[McKenna 2010] \
*GATK-gCNV:* Babadi M, Fu JM, Lee SK, Smirnov AN, Gauthier LD, Walker M, Benjamin DI, Zhao X, Karczewski KJ, Wong I, Collins RL.
GATK-gCNV enables the discovery of rare copy number variants from exome sequencing data.
Nature genetics. 2023 Sep;55(9):1589-97. #c[Babadi 2023] \
*Delly:* Rausch T, Zichner T, Schlattl A, Stütz AM, Benes V, Korbel JO. DELLY: structural variant discovery by integrated paired-end and split-read analysis.
Bioinformatics. 2012 Sep 15;28(18):i333-9. #c[Rausch 2012]

#v(0.6em)

*Picard:* "Picard Toolkit." 2019. Broad Institute, GitHub Repository.
#link("https://broadinstitute.github.io/picard/"); Broad Institute.
#c[Broad Institute 2019]

#v(0.6em)

Hoeffding (1948), _A class of statistics with asymptotically normal distribution_, Ann. Math. Statist. 19(3):293-325 -- the original decomposition. \
Sobol' (1993), _Sensitivity estimates for nonlinear mathematical models_, Math. Model. Comput. Exp. 1(4):407-414 -- the variance indices. \
Saltelli et al. (2008), _Global Sensitivity Analysis: The Primer_ -- the standard practitioner treatment. \
Owen (2013), _Variance components and generalized Sobol' indices_, SIAM/ASA J. Uncertain. Quantif. 1(1):19-41 -- the tidiest modern statement.

#set text(size: 11pt)
#set par(justify: true)

== Notes

Use #link("https://www.sciencedirect.com/science/article/pii/S0888754324001836?via%3Dihub#s0110") for PennCNV justification \
CNV detection tools from NGS data: #link("https://pmc.ncbi.nlm.nih.gov/articles/PMC12218993/") \
2020 detection tool comparison paper: #link("https://pmc.ncbi.nlm.nih.gov/articles/PMC7059689/") \
WES and WGS paper: #link("https://www.tandfonline.com/doi/full/10.1586/14737159.2015.1053467")
