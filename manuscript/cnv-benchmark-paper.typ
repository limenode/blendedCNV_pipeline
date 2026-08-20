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
BGE data has also shown to be a promising method to capture un-biased genetic diversity in underrepresented populations at a fraction of the cost of deep WGS, with performance that is competitive against population-specific GWAS arrays #c[Boltz 2026].

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

== Sequence-based CNV Calling

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

== Benchmark Dataset Preparation

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

== BED File Conversion

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

== CNV Overlap and Adjacency Graph Building

After retrieving all CNV call sets of interest and converting them to a standardized BED format, the CNV data was loaded into Python to generate a graph representation of call sets.
Individual graphs were generated for 1) All benchmark calls, 2) Sequence-based CNV calls per tool, 3) Sequence-based CNV calls aggregated across all tools, and 4) SNP array CNV calls.

The edge calculation process began by generating an iterable list of nodes and ensuring that the input is chromosome and start sorted.
An empty connected component is initialized for each unique partition of tuple (Sample ID, Chromosome, SV type), defining the boundary across which edges cannot be made.
For each sequential node, it checks whether or not there exists overlap between itself and any nodes in the appropriate partition's connected component: If there is, create an "overlap edge" between nodes with the reciprocal overlap as the weight; otherwise, create a "gap edge" between the nodes with their distance from each other as the weight and reset the connected component to only contain the current node.
This continues until all edges are built.

== Consensus CNV Calling

Merging sub-networks of the graphs according to different edge and network filtering parameters was also performed in order to derive consensus CNV call sets.
Edges were filtered by minimum reciprocal overlap threshold or by maximum distance between calls, in a mutually exclusive manner.
Once the edges were filtered, all connected components were retrieved and additionally filtered for a minimum number of unique sources across the nodes in each component.
Once all connected components that passed the filters were retrieved, the nodes within each component were aggregated by union into a single child call node, taking the minimum start position and the maximum end position across all member nodes.

Consensus CNV call graphs were generated for the benchmark sets to create a single merged benchmark set to serve as the primary truth source for downstream analysis.
Consensus call graphs were also generated for the three sequence-based call sets derived from the calling tools of interest.
Multiple versions of these two consensus call graphs were generated across different filtering parameters for downstream analysis of performance across the parameter field.

== Null Model for Caller Agreement

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

== Size Floor Filtering

All three sequence-based callers were run with a 1 kb bin size and consequently cannot resolve CNVs below that width, whereas the merged benchmark is dominated by events an order of magnitude smaller.
Comparing the two sets without a size restriction therefore charges the callers with failing to detect variants that the chosen bin size places outside their resolution, which depresses recall for a reason that has nothing to do with sequencing depth.

A size floor was applied to the query and truth call sets *symmetrically*, that is, both sides were restricted to intervals of at least the floor before any matching was performed.
Filtering only the query side would leave sub-floor benchmark intervals in the denominator of recall and move the attainable maximum for a non-biological reason; filtering only the truth side would do the reverse to precision.

The floor was fixed at 1 kb on physical grounds, being the bin size common to all three callers, and was chosen before any performance metric was consulted.
To confirm that this choice is not merely convenient, the floor was swept over \[1 bp, 100 kb\] at 80 logarithmically spaced points, with precision, recall, and F1 evaluated at every point for each of the six 30x query call sets.
Alongside the metrics we tracked the number of intervals surviving in each call set and the maximum attainable recall, defined as the ratio of query calls to truth intervals and capped at one.
That ratio is the largest recall achievable if every query call were to match a distinct truth interval, so it bounds recall from above independently of how well the callers perform, and it identifies the regime in which a call set has become larger than the truth set it is being scored against.

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

Various graphs were generated across all tested binary classification metrics using the `matplotlib` Python package.
Generated plots include density distributions showing the probability density of CNVs across different CNV sizes, cumulative distribution functions (CDFs) showing the proportion of CNVs ≤ each size threshold, and complementary cumulative distribution functions (CCDFs) showing the proportion of CNVs ≥ each size threshold.
Additionally, Venn diagrams and plots for the distribution of query calls that had one-to-many relationships with truth calls and vice versa were also created.

Figures were generated with matplotlib \[v3.11.1\].
Three families of plots summarise the classification.
Size-resolved performance is shown as kernel-smoothed density curves over CNV size, together with cumulative and complementary cumulative distribution functions giving the proportion of CNVs at or below, and at or above, each size threshold.
Query-to-truth matching structure is summarised by the number of query calls matching more than one truth call and vice versa; because matching is many-to-many, true-positive counts on the query and truth sides do not necessarily coincide.
Finally, each of the three primary pipeline parameters -- the query consensus reciprocal-overlap threshold, the benchmark padding, and the classification reciprocal-overlap threshold -- was varied, with the effects of variation plotted against precision, recall, and F1 score and against the number of calls that enter the comparison.
These one-at-a-time analysis profiles motivate the parameter ranges used in the joint analysis below.

== Variance-based sensitivity analysis and Pareto front

=== Parameter Sweep

The three parameters of interest were varied jointly over a factorial grid: the query consensus reciprocal-overlap threshold and the classification reciprocal-overlap threshold each over \[0.05-0.95, in steps of 0.05\], and the benchmark padding over \[0, 10, 25, 50, 100, 200, 400, 700, 1000\].
Cumulative precision, recall, and F1 were evaluated at every setting combination, together with the number of query and truth calls entering each comparison.

Two bounds require justification.
Padding is applied to both interval ends, so a padding of $p$ increases an interval's span by $2p$; with a median benchmark interval of 144 bp (IQR of \[66-355\] bp), padding of 1 kb already inflates the median interval approximately 14-fold.
Padding was therefore capped at 1 kb, beyond which the operation no longer represents tolerance for boundary imprecision, and instead progressively fuses neighboring distinct benchmark intervals (\[\~546k calls unpadded vs. \~464k calls at 1kb padding vs. 395k at 10kb padding).
Second, a reciprocal-overlap threshold of exactly zero permits calls with an overlap of at least 1 bp to be admitted as true positive.
This produces distinct results from the range of reciprocal overlap threshold values tested and is excluded from the main grid, but included in the full grid for which analysis is shown in the supplementals.

=== Sensitivity Indices

The contribution of each parameter was quantified by variance-based sensitivity analysis (Sobol', 1993).
Writing $Y$ for a metric (precision, recall, f1) and $x_1, x_2, x_3$ for the parameters, the first order Sobol index $S_i$ of parameter $i$ is the fraction of the metric's variance attributable to that parameter acting alone.

$ S_i = ("Var" (EE [Y | X_i])) / ("Var" (Y)) $

The total-order index $S_(T i)$ additionally includes every interaction involving $i$.

$ S_(T i) = 1 - ("Var" (EE [Y | bold(X)_(tilde i)])) / ("Var" (Y)) $

The difference $S_(T i) - S_i$ is the variance a parameter contributes only jointly with others.
Because the design is a complete factorial grid, each conditional expectation is a marginal mean over the grid and both indices were computed exactly.
The complete decomposition, comprising all first-, second-, and third-order terms, sum to one, and this was verified numerically.

=== Additivity

The same decomposition expresses the metric field as a grand mean plus one univariate function per parameter:

$ Y approx mu + sum_i f_i (X_i) wide "with" wide f_i (x) = EE [Y | X_i = x] - mu $

The coefficient of determination of this additive model equals the sum of the first-order indices, and quantifies the extent to which the one-parameter profiles (See Figures A-B) describe the joint behavior of the pipeline.

=== Dependence on the swept ranges

Sobol indices are variance ratios with respect to a distribution over the inputs.
They therefore describe sensitivity within the examined region and are not invariant to the choice of that region.
As such, indices were additionally computed over an intentionally over-wide grid extending to $10^6$ bp (Supplementary Table X) in order to compare physical arguments for the previously described parameter boundaries with empirical data.

=== Pareto front

Because precision and recall span very different ranges across the grid, F1 is close to a monotone function of recall alone and obscures the trade-off between the two.
Parameter settings were therefore also summarised by their Pareto front: a setting is dominated if another attains at least equal precision and recall and strictly exceeds it on one, and the front comprises the non-dominated settings.

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

== Input Call Sets After Parsing

All input call sets were normalised to a common per-sample BED representation and every call set was restricted to the thirteen samples of interest to ensure no statistic reflected differences in cohort composition.
Liftover was performed on the calls derived from the SNP Array from hg18 to hg38/GRCh38 to align coordinates with all other call sets.
Table 1 summarises the resulting call sets: the three sequence-based callers at each of the four coverages, the SNP array control, and each of the three benchmark sources.
Counts are given after removal of calls overlapping poorly mappable regions by at least 1% of their total length.
Pre-filter counts and the full liftover accounting are given in Supplementary Table S1, and the complete exclusion accounting -- bases removed against bases actually inside the mask, and the split by CNV type -- in Supplementary Table S2.
The results from different stringencies for overlap with poorly mappable regions were tested and are given in Supplementary Table S3.
1% was determined to be the most reasonable choice for downstream analysis on the physical grounds that increasing overlap percentage would permit more calls with unreliable breakpoints and decreasing overlap requirements would more often cause true calls to be removed; emperical results validated this.

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
The call sets that enter the downstream analysis are equivalent in that both have had these regions removed, but the read-depth normalisation performed internally by CNVpytor and GATK-gCNV was not carried out over the same genomic territory in the two cases.
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

== Sequence-based Consensus Call Set Construction

All outputs from the sequence-based consensus call sets were aggregated into a single graph and edges between overlapping and adjacent calls were computed accordingly.
Analysis of the consensus construction was required in order to determine the general behavior of each of the callers and if consensus analysis was a suitable approach to retrieve an informative population of calls.
We therefore characterised the three callers relative to each other at 30x, where the evidence available to them is greatest, and then followed the same quantities down through the reduced coverages.

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
Whether that private component consists of artefacts or of genuine calls beyond the reach of the other callers cannot be settled from agreement alone, and is taken up in the comparison against the benchmark.
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
The 2x 1/3 call set is therefore both the second largest of the four and the one that loses the most to any agreement requirement, which suggests that much of it consists of caller-specific artefacts arising from reduced confidence at low coverage rather than of calls that the remaining callers failed to reach.

#v(0.3em)

Refitting the null model at each coverage shows that the two populations respond to depth in opposite ways (Table 4).
For the concordant population, the fitted core's total events (predicted number of calls given the null model holds) falls from 5,738 at 30x to 1,436 at 2x, and the detection rates per caller fall alongside it, from 0.79 to 0.59 for CNVpytor, from 0.72 to 0.45 for Delly, and from 0.73 to 0.46 for GATK-gCNV.
The two contribute in similar measure, since holding the rates at their 30x values while shrinking the core to its 2x size would leave 1,210 concordant components against the 721 observed.
The caller-private population does not follow.
The ratio of private to concordant components consequently rises as coverage decreases, from 4.0 at 30x to 25.6 at 2x.
This trend holds despite the total number of private calls not demonstrating a clear trend in depth, with call count being larger at 2x than at 4x.
This is evidence to support the proposed mechanism behind the losses to the agreement requirement described above: what additional depth supplies is primarily agreement between callers, while private calls from a caller are produced about as readily at 2x as at 30x.
Whether that private population is artefactual cannot be settled from its coverage response alone, although a population whose size is largely independent of the evidence available to produce it is difficult to attribute to the underlying genetic data.
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


== Choosing a Detectable Size Domain

The merged benchmark contains 178,838 intervals across the thirteen samples, with a median size of 145 bp; only 13.9% reach 1 kb and 2.8% reach 10 kb.
The six 30x query call sets -- as shown in Figure 4 -- are between 7 and 75 times smaller and, because every caller was run at a 1 kb bin size, hold almost nothing below that width.
Scoring the query sets against the benchmark without a size restriction therefore primarily measures the mismatch in resolution between the benchmark and the callers rather than an effect of sequencing depth, which is what this study set out to isolate.
We swept a size floor applied symmetrically to both sides and examined how the comparison behaves as the size floor rises (Figure 4).

#figure(
  image("/results/size_floor/detectable_size_domain_pub.png", width: 100%)
)

#cap("Figure 4:")[
  Effect of a size floor applied symmetrically to 30x WGS-derived query call sets and the merged benchmark call set, swept over \[1 bp, 100 kb\] at 80 logarithmically spaced points.
  (A) Number of intervals surviving the floor in each call set, with the merged benchmark as a dashed black line.
  (B) Recall, with the maximum attainable recall (query calls divided by truth intervals, capped at one) as a dashed line of the same colour.
  (C) Precision.
  (D) F1, with each call set's maximum marked.
  The shaded band marks 1--5 kb, spanning every F1 maximum; the dashed vertical line marks the 1 kb floor adopted for all subsequent analyses.
  In panels C and D, each curve is drawn only while the call set behind it retains at least 100 intervals, since precision estimated from a few dozen calls is not comparable with precision estimated from thousands.
  The merged benchmark is 99.8% deletions, so these curves primarily describe deletion detection.
]

#v(0.8em)

Three features of this sweep together identify the usable domain.

First, the benchmark loses intervals far faster than any query call set.
Below roughly 500 bp the truth set outnumbers the largest query set by an order of magnitude, but it falls below the 1-of-3 consensus set above a 1,460 bp floor and below CNVpytor above 2,616 bp (Figure 4A).
Above those points the comparison has inverted: the callers report more CNVs than the benchmark contains, and precision is bounded by the size of the truth set rather than by the accuracy of the calls.

Second, recall is constrained by a ceiling that is a property of the two call set sizes rather than of detection (Figure 4B).
At an unrestricted floor (size floor = 0 bp) the maximum attainable recall is 0.135 for the 1-of-3 set and 0.013 for the 3-of-3 set, so even a caller that matched a distinct benchmark interval with every single one of its calls could not exceed those values.
Raising the floor lifts the ceiling for every call set, but it does so unevenly: the 1-of-3 and CNVpytor sets saturate at 1.0 above floors of 1,460 bp and 2,616 bp respectively, while Delly, GATK-gCNV, and the 2-of-3 and 3-of-3 consensus sets never approach it, reaching maxima of 0.55, 0.36, 0.31, and 0.18.
Recall is therefore interpretable as a detection measurement only below the point at which a given call set saturates.

Third, precision is flat from 1 bp to approximately 1 kb for every call set and declines above it (Figure 4C).
The flat region is the direct consequence of the bin size: over that range the floor removes benchmark intervals almost exclusively, because the callers had produced essentially nothing there to remove, so the query sets and their precision are unchanged while the truth set falls from 178,838 intervals to 24,820.
Above 1 kb the floor begins to remove query calls as well, and precision falls for every set, steeply for CNVpytor and the 1-of-3 consensus (0.318 to 0.052 and 0.317 to 0.051 between the unrestricted case and a 100 kb floor) and more gradually for GATK-gCNV, which is the most size-stable of the callers (0.566 to 0.448 over the same range, the upper end of which lies beyond the 64.6 kb floor at which its curve is cut for low counts).

Taken together, these place the usable domain immediately above the bin size.
We fixed the floor at 1 kb, chosen on the physical grounds that no caller in this study can resolve a CNV narrower than its bin.
The sweep supports that choice: F1 reaches its maximum between 1,689 bp and 4,051 bp for all six call sets, with the 2-of-3 consensus highest at 0.348 (Figure 4D).
A 1 kb floor therefore sits just below the empirical optimum for every call set simultaneously.

This floor is applied to every call set and to the benchmark for all analyses that follow, at all four coverages.
It was selected using 30x data only, which is the arm with the finest resolution and therefore the most permissive: a floor set there admits calls at 2x that fall below the resolution attainable at that depth, biasing the comparison against the low-coverage hypothesis rather than in its favour.

== CNV Size Distribution Characteristics

#figplaceholder("image9 — Figure 3, KDE of CNV size distributions at 30x")

#cap("Figure 3:")[
  Kernel density estimates for CNV size distributions of call sets derived from 13 30x WGS samples, accompanied by corresponding statistics for the control SNP microarray CNV call set and the benchmark call set.
  The x-axis is presented with a log-10 scale for better trend visualization.
  CNV sizes are bounded by the analysis window range of \[500 bp, 1 Mb\].
]

#v(0.8em)

Next, we examined the size distributions of the CNV callsets.
CNVpytor did not call any CNVs below 2000 bp when using a 1kb window as a consequence of requiring two adjacent bins to have high confidence for the same CNV type to call a CNV.
On the other hand, GATK-gCNV called CNVs of sizes equal to the bin size, and Delly called CNVs lower than the bin size with its more sophisticated breakpoint prediction method.
Furthermore, compared to all other call sets, CNVpytor had a significantly higher median and IQR, being nearly 2-fold higher than the next highest in each category (Supplemental Table Size Distribution Statistics).

Although size distributions for the individual callers were strongly dependent on coverage, each caller demonstrated different correlations.
At 30x coverage, CNVpytor had a slightly uniform distribution across the interval \[2kb, 100kb\] (Figure 3).
As coverage decreased, far less calls were made between \[2kb, 10kb\] while a sharp density peak increased dramatically at 50kb (Supplemental Figure Per Caller Size Distribution Graphs A).
GATK-gCNV, unlike the other callers, had a clear tri-modal distribution at 30x, with the first two peaks (1kb, 2kb) decreasing in density and the third peak (5-6kb) widening as the coverage decreased (Supplemental Figure x for Per Caller Size Distribution Graphs B).
The distribution from Delly was nearly uniform across the interval \[1kb, 10kb\] across all coverages, with the only notable trend being that the density at the tail end of the \[1kb, 10kb\] slightly decreased at each lower coverage step (Supplemental Figure Per Caller Size Distribution Graphs C).

For the consensus call sets at 30x coverage, when consensus stringency increased distributions went from relatively uniform across most of the analysis window to having a well defined peak at 6 kb (Figure 3).
Moving to lower coverages, the consensus call set distributions become more bell shaped and symmetrical around the 10kb region.
Particularly for the 4x and 2x coverages, some of the peak density regions from the single callers start to propagate into the consensus call set resulting in a multimodal distribution (Supplemental Figure Consensus Calling Distribution Graphs).

The SNP Array callset showed a nearly symmetrical distribution centered around 6kb.
In contrast to the candidate call sets, the benchmark set was strongly right-skewed, with most benchmark CNVs in the 500 bp--10 kb range and a minor density peak around \~6 kb.
There is a significant drop in benchmark calls past \~7kb (Figure 3).

The consensus-based call sets revealed a strong dependence between median CNV size and consensus stringency, with higher stringency generally yielded higher median CNV size.
The only exception is the 30x coverage, where the median size stays relatively constant among the consensuses (Supplemental Table Size Distribution Statistics).
This trend is consistent with expectations that larger CNVs would have greater confidence in identification and thus greater agreement between callers compared to smaller CNVs, especially at lower coverages.

== CNV Calling Performance Statistics

#v(0.5em)

#block(width: 100%)[
#set text(hyphenate: false, size: 10pt)
#table(
  columns: (1fr, auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right, right, right),
  table.header(
    [], [True Positives], [False Positives], [False Negatives\*],
    [Precision], [Recall\*], [F1 Score\*], [F1/2 Score\*], [F2 Score\*],
  ),
  [*30x Coverage -- CNVpytor*], [5592], [12105], [3120], [0.316], [0.642], [0.423], [0.352], [0.532],
  [*30x Coverage -- GATK-gCNV*], [4971], [3591], [3741], [0.581], [0.571], [0.576], [0.579], [0.573],
  [*30x Coverage -- Delly*], [4398], [2792], [4314], [0.612], [0.505], [0.553], [0.587], [0.523],
  table.hline(stroke: 0.3pt),
  [*30x Coverage -- 1/3 Consensus*], [8201], [17521], [511], [0.319], [0.941], [0.476], [0.367], [0.677],
  [*30x Coverage -- 2/3 Consensus*], [4308], [715], [4404], [0.858], [0.494], [0.627], [0.748], [0.540],
  [*30x Coverage -- 3/3 Consensus*], [2302], [77], [6410], [0.968], [0.264], [0.415], [0.631], [0.309],
  table.hline(stroke: 0.3pt),
  [*SNP Array*], [541], [1068], [8171], [0.336], [0.062], [0.105], [0.179], [0.074],
)
]

#cap("Table 5:")[
  Binary classification metrics of CNV call sets against benchmark CNV call, which are derived from the following datasets: 1000G phase 3, HGSVC3, and ONT Vienna.
  Metrics for each call set were calculated from the CNVs of sizes that fall within the the analysis window of \[500 bp, 1 Mb\].
  A true positive indicates that a CNV in the call set had at least a 50% reciprocal overlap with one of the CNVs in the benchmark call set.

  \*Due to the significant differences in call set size between the benchmark set and the evaluated call sets, the false negative value only considers CNVs that were discovered by at least one of the call sets tested.
  This ensures that the statistics derived from the false negative count (Recall and F Scores) remain interpretable and highlight the comparisons that are critical to this study.
]

#v(0.8em)

Next, we compared the callsets with the benchmark set to evaluate their performance.
Many distinct patterns emerge when comparing the different call sets that were all derived from 30x coverage WGS data.
Among the callers, CNVpytor had higher recall and lower precision compared to the other single caller CNV call sets, and overall had slightly lower performance than the others according to the F1-score.

When compared to the consensus call sets, single caller performance was middling compared to the consensus.
Consensus calling for high-coverage resulted in dramatic increases in specific performance metrics, largely corroborating our preliminary discoveries and those found in other studies.
The 1/3 consensus call sets had dramatically higher recall and lower precision, consistent with the expectation that permissive strategies across callers increase sensitivity but also propagate caller-specific artifacts and inflate the false discovery rate #c[Ho 2020] #c[Liu 2022].
Conversely, the 3/3 consensus call sets had higher precision and low recall, clearly reflecting a stringent agreement strategy that improves confidence at the cost of sensitivity and excludes variants that are detectable but not consistently recovered across algorithms #c[Ho 2020] #c[Liu 2022].
The 2/3 consensus calling method demonstrated a promising performance balance compared to the other two consensus types.
This set had a dramatic improvement in precision compared to the 1/3 set, nearing the performance of the 3/3 set, while still maintaining a decent improvement in recall compared to the 3/3 set.
The F1 score highlights the 2/3 consensus set as the set with highest overall performance with the best balance in recall and precision among the consensus call sets and against the single callers.

The SNP array has poor performance when compared to all sequence-based call sets.
While the precision slightly exceeds that of CNVpytor and the 1/3 consensus sets, the recall and all F-score metrics are much worse.
It is important to note that this could be a result of the benchmark sets containing CNVs that are undetectable with the Omni-2.5 Array kit, resulting in inflation of the False Negative count and a strong bias for the recall and F-scores of the sequence-based call sets.

Overall, the performance of the call sets that we had evaluated demonstrated the improvement in performance from consensus calling over single caller approaches and the performance balance offered by the 2/3 consensus calls.
This largely agreed with patterns highlighted by various paradigms established in prior studies, and therefore we evaluated the performance of the 2/3 consensus calls in our downstream analysis.

== Size Distribution and Performance Statistics of 2/3 Consensus Call Sets across Coverages

#v(0.5em)

#block(width: 100%)[
#set text(hyphenate: false, size: 10pt)
#table(
  columns: (1fr, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right),
  table.header(
    [Call Set], [Total Call Count], [Min. Size], [Median Size], [Max. Size], [IQR],
  ),
  [*30x Coverage -- 2/3 Consensus*], [5,048], [654], [5000], [669175], [6000.00],
  [*6x Coverage -- 2/3 Consensus*], [1781], [898], [10485], [868000], [24000.00],
  [*4x Coverage -- 2/3 Consensus*], [1147], [1000], [17000], [908000], [34412.00],
  [*2x Coverage -- 2/3 Consensus*], [745], [1550], [26000], [969135], [46000.00],
  table.hline(stroke: 0.3pt),
  [*SNP Array*], [1,609], [500], [8127], [776884], [17966.00],
  [*Benchmark*], [89,565], [500], [1441], [961615], [2824.00],
)
]

#cap("Table 6:")[
  CNV call set size statistics of 2/3 consensus call sets derived from 13 WGS samples of coverages across 30x, 6x, 4x, and 2x.
  This is accompanied by corresponding statistics for the control SNP microarray CNV call set and the benchmark call set.
  The IQR is color mapped with the higher values in red and the lower values in green.
  CNV sizes are strictly bounded by the analysis window range of \[500 bp, 1 Mb\].
]

#v(0.8em)

#figplaceholder("image10 — Figure 5, KDE of 2/3 consensus sizes across coverages")

#cap("Figure 5:")[
  Kernel density estimates for CNV size distributions of 2/3 consensus call sets derived from 13 WGS samples of coverages across 30x, 6x, 4x, and 2x.
  This is accompanied by corresponding statistics for the control SNP microarray CNV call set and the benchmark call set.
  The x-axis is presented with a log-10 scale for better trend visualization.
  CNV sizes are strictly bounded by the analysis window range of \[500 bp, 1 Mb\].
]

#v(0.8em)

We established that the consensus calls performed substantially better than any single CNV calling tool, and the 2/3 consensus calls are the best balance between the performance metrics.
Therefore, we compared the 2/3 consensus calls across coverages to evaluate the performance of lcWGS-based CNV calls from a BGE sequencing pipeline.

The number of CNV calls decreases as the coverage is lowered.
Compared to 5,048 CNVs in the 30x coverage consensus set, 6x coverage set has about one-third CNVs (1,781), while 2x coverage set only has 745 CNVs (Table 6).
In addition, there is a strong correlation between coverage and variability.
As coverage decreases, IQR increases dramatically (Table 6) and the density of calls spreads out wider across the analysis window (Figure 5).
Additionally, both the median and the peak of the size distribution curve shift to the right as coverage decreases, corresponding with expectations that lower coverage will lose more calls in the few-to-several kilobase range due to the lack of genomic evidence.
The benchmark call set has a significantly lower IQR, but this is largely attributed to the fact that the benchmark has most of its density in the several hundred base regions, which is nearly undetectable by the sequence-based callers using a 1kb bin size.
The SNP array has an IQR and median slightly lower than those of the 6x coverage, while additionally having a similar size distribution to that of the lower coverage sequence-based call sets.
This may indicate similarities in the genomic size ranges in which these call sets have their highest confidence CNV calls despite using different sources of genetic evidence.

#figplaceholder("image11, image12, image13 — Figure 6, three benchmark-recall Venn diagrams", height: 6cm)

#cap("Figure 6:")[
  Venn diagrams of the identification of CNVs in the benchmark set by 30x coverage-based 2/3 consensus calls, SNP microarray-based calls, and one of three lower coverage-based consensus calls out of 6x, 4x, and 2x.
  The bottom shows the total number of CNVs in the benchmark set, the number and percentage of CNVs detected by at least one of the methods, and the number of CNVs not detected by any of the listed methods.
  Written below each category label is the total count of CNVs identified by said category and the associated percentage of the total detected CNVs.
  Categories that yielded very low counts (i.e. \<10 calls) are not shown in the Venn diagram for clarity, and account for any discrepancies between the numbers in the Venn diagram and the totals that are shown.
]

#v(0.8em)

Next, we compared each low-coverage 2/3 consensus call set individually against the high-coverage consensus set and the SNP array to quantify how well different coverages capture CNVs that are detectable at high-coverages and by the SNP array.
High-coverage sequencing calls accounted for the majority of recalled benchmark CNVs, and contained significant proportions of CNVs recalled by both the SNP Array and each of the lower coverage call set categories (Figure 6).
As the coverage decreased, the overlap between the 30x and low-coverage call sets rapidly decreased, with a near two-fold decrease shown when moving to each low-coverage category (1066 → 577 → 269 for 6x → 4x → 2x).
The overlap between the low-coverages and the SNP array also exhibited a similar decrease, with a larger proportional drop in overlap observed going from 4x to 2x coverage compared to when going from 6x to 4x coverage.
Additionally, while the number of total CNVs uniquely identified by the low-coverage category decreased with the coverage, the percentage of CNVs uniquely identified steadily increased, with the results for 6x, 4x, and 2x being 4.4%, 5.6%, and 7.1% respectively.

#v(0.5em)

#block(width: 100%)[
#set text(hyphenate: false, size: 10pt)
#table(
  columns: (1fr, auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right, right, right),
  table.header(
    [Call Set], [True Positives], [False Positives], [False Negatives\*],
    [Precision], [Recall\*], [F1 Score\*], [F1/2 Score\*], [F2 Score\*],
  ),
  [*30x Coverage* -- 2/3 Consensus], [_4295_], [_753_], [_4417_], [_0.851_], [_0.493_], [_0.624_], [_0.743_], [_0.538_],
  [*6x Coverage* -- 2/3 Consensus], [_1395_], [_386_], [_7317_], [_0.783_], [_0.160_], [_0.266_], [_0.440_], [_0.190_],
  [*4x Coverage* -- 2/3 Consensus], [_833_], [_314_], [_7879_], [_0.726_], [_0.096_], [_0.169_], [_0.313_], [_0.116_],
  [*2x Coverage* -- 2/3 Consensus], [_420_], [_325_], [_8292_], [_0.564_], [_0.048_], [_0.089_], [_0.180_], [_0.059_],
  [*SNP Array*], [_541_], [_1068_], [_8171_], [_0.336_], [_0.062_], [_0.105_], [_0.179_], [_0.074_],
)
]

#cap("Table 7:")[
  Binary classification metrics of 2/3 consensus CNV call sets across all coverages and the SNP Array CNV call set.
  Metrics for each call set were calculated from the CNVs of sizes that fall within the the analysis window of \[500 bp, 1 Mb\].
  A true positive indicates that a CNV in the call set had at least a 50% reciprocal overlap with one of the CNVs in the merged benchmark call set.

  \*Due to the significant differences in call set size between the benchmark set and the evaluated call sets, the false negative value only considers CNVs that were discovered by at least one of the call sets tested.
  This ensures that the statistics derived from the false negative count (Recall and F Scores) remain interpretable and highlight the comparisons that are critical to this study.
]

#v(0.8em)

#figplaceholder("image14, image15, image16 — Figure 7, density of precision / recall / F1", height: 6cm)

#cap("Figure 7:")[
  Probability density graphs for the precision, recall, and F1-score of CNVs identified from all 2/3 consensus call sets across all coverages.
  100 log-spaced interval points between 500bp and 1 Mb were used to estimate the probability distribution shown, and the values of the points are visualized as transparent points scattered throughout the graph with colors corresponding to the respective call set.
  The curves shown were derived from Gaussian kernel smoothing (σ=5.0) of these 100 points.
]

#v(0.8em)

#figplaceholder("image17, image18, image19 — Figure 8, cumulative precision / recall / F1", height: 6cm)

#cap("Figure 8:")[
  Cumulative distribution graphs for the precision, recall, and F1-score of CNVs identified from all 2/3 consensus call sets across all coverages.
  100 log-spaced interval points between 500bp and 1 Mb were used to estimate the probability distribution shown, and the values of the points are visualized as transparent points scattered throughout the graph with colors corresponding to the respective call set.
  The curves shown were derived from Gaussian kernel smoothing (σ=5.0) of these 100 points.
]

#v(0.8em)

Finally, we evaluated the performance of each 2/3 consensus call set across size ranges.

The overall classification performance of the 2/3 consensus call sets (Table 7) reveals a clear performance hierarchy across coverage depths.
At 2x coverage, performance was near-identical to the SNP array: the 2x call set achieved substantially higher precision (0.564 vs. 0.336), while the SNP array modestly outperformed it in recall (0.062 vs. 0.048).
Across all F-scores, the two approaches had marginal differences, with the F₁ and F₂ scores favoring the SNP array and the F1/2 score being near identical between 2x and SNP Array (0.180 vs. 0.179).
Beyond 2x coverage, sequencing-based call sets consistently and substantially outperformed the SNP array across all metrics.
The 4x coverage call set showed meaningful gains in precision (0.726) and F1/2 score (0.313) relative to the 2x coverage call set, and the 6x coverage call set saw further improvements across all performance metrics.
The 30x coverage call set demonstrated the strongest overall performance, with precision of 0.851 and a substantially higher recall (0.493) than all lower-coverage and array-based call sets.

Peak precision for all sequencing-based call sets except 2x approached \~90% in the several-kilobase range (Figure 7).
Precision trends across coverage levels converged past the size interval where each lower-coverage call set reaches its peak, such that 30x, 6x, and 4x all yielded nearly identical precision past 10 kb, with 2x only slightly lower through the 10--50 kb range before also converging.
The SNP array-based call set only exceeded 4x and 2x in precision within small CNV size ranges of \[500bp, 1500bp\] and \[500bp, 5000bp\], respectively.
The shape of the precision trend for the SNP array additionally demonstrates similarities to that of the 2x coverage call set past 10 kb (Figure 7).

Recall exhibited a pronounced dependence on both coverage and CNV size, with the 30x coverage call set substantially outperforming all others and was the only call set to demonstrate a distinct peak in the several-kilobase range.
Decreasing coverage resulted in flatter recall densities and right-shifted distribution, with their highest performance in the 20 kb to 300 kb range (Figure 7).
An exception to this trend is with the 6x coverage call set with a unique recall density peak at \~400kb.
The SNP array distribution exhibited the lowest recall density across nearly the entire analysis window (Figure 7).

A consistent 1.5- to 2-fold increase in cumulative recall was observed when moving from 2x to 4x coverage and again from 4x to 6x past 10 kb (Figure 8).
The SNP array's cumulative recall exceeded that of 2x coverage across the entire analysis window and very slightly exceeded that of 4x coverage below 7 kb.

The F1 score is largely dominated by the trends of the recall of the call sets, with the 30x coverage being the only set to demonstrate significant performance in smaller size regimes around 1 kb.
Decreasing coverage results in lower peak performance density, densities further shifted right, and more uniform densities.
Considering cumulative performance, the SNP array outperforms 4x coverage until 5-6 kb CNV sizes and outperforms 2x coverage until 50 kb (Figure 7).

Across all methods, deletions yielded higher performance in all metrics compared to duplications.
Duplications additionally demonstrated poorer performance at the ends of the 500bp to 1Mb analysis window (Supplemental Figure SV Types F1-Score Graphs).

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

#pagebreak()

= Leftovers

#emph[
  The remaining sections are working notes carried over from the draft, not manuscript text.
]

== Comments

Comment on BGE: \
Add a few sentences describing how the data is being used (imputation) with some references. \
Then leads into using BGE for CNV detection, cites individual papers, then proposes the unknown: "However, the performance of lcWGS-based CNV detection is unknown...".
See the 2nd to last paragraph of the nanopore SV paper (#link("https://link.springer.com/article/10.1186/s13059-019-1858-1")) for an example.

were determined to perform well with our lcWGS data while maintaining a significant potential for scaling up to larger sample sizes and different coverages.

A majority of the benchmark set calls were smaller structural variations, and filtering all the call sets for CNVs within the 500 bp--1 Mb window significantly reduced the number of calls in the benchmark set while incurring minimal to no losses in CNV calls from the evaluated sets.
Out of the evaluated call sets, the SNP Array-based call set experienced the greatest difference when restricting to the analysis window, with the loss of 51 calls (3.1%).

== Abstract

#link("https://pubmed.ncbi.nlm.nih.gov/31076843/") #c[Lauer 2019] \
An evolving view of copy number variants

WGS has shown to improve disorder diagnosis by a significant margin over WES, but performing WGS incurs a significant additional sequencing cost #c[Ewans 2022].

After performing CNV calling across all of the coverages included in this study, we performed a post-processing step and removed artifactual calls made in genomic regions that were previously excluded from the input sequences due to their poor mapability (See Supplementals \[Excluded Regions Graph\]).

== Counts

Requiring support by at least one callers greatly increased the number of reported CNVs (e.g. 30x coverage: 33,449 for 1/3 vs 5,048 for 2/3; 6x coverage: 19,476 total calls for 1/3 vs 1,781 for 2/3), consistent with the expectation that permissive strategies across callers increase sensitivity but also propagate caller-specific artifacts and inflate the false discovery rate #c[Ho 2020] #c[Liu 2022].

In contrast, requiring 3/3 caller agreement produced substantially smaller call sets (e.g., 6x coverage: 703 total calls; 30x coverage: 2,464 total calls), reflecting a stringent agreement strategy that improves confidence at the cost of sensitivity and excludes variants that are detectable but not consistently recovered across algorithms #c[Ho 2020] #c[Liu 2022].

#v(0.5em)

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  table.header(
    [], [1/3 Consensus], [2/3 Consensus], [3/3 Consensus],
  ),
  [*30x Coverage*], [25872], [5048], [2381],
  [*6x Coverage*], [19476], [1781], [678],
  [*4x Coverage*], [15240], [1147], [284],
  [*2x Coverage*], [19740], [745], [159],
)

#cap("Table 2:")[
  CNV call set counts for all consensus-based call sets across coverages with different caller agreement requirements.
  Counts are color mapped with higher values in green and lower values in yellow.
  Call sets only include CNVs that fall within the analysis window of \[500 bp, 1 Mb\].
]

#v(0.8em)

The only deviation from this trend described above is with the 30x coverage, and in general as the coverage decreases the difference in median between consensuses increases.
This aligned well with the following expectation: at higher coverage, callers have higher agreement between each other particularly in the several kilobase regime as a result of having more substantial genomic evidence for calling small CNVs and/or operating at coverages that these tools were designed to excel with.

#v(0.5em)

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  table.header(
    [], [_1/3 Consensus_], [_2/3 Consensus_], [_3/3 Consensus_],
  ),
  [_30x Coverage_], [_6000_], [_5000_], [_5000_],
  [_6x Coverage_], [_8000_], [_10485_], [_11042.5_],
  [_4x Coverage_], [_15000_], [_17000_], [_22000_],
  [_2x Coverage_], [_9000_], [_26000_], [_39000_],
)

#cap("Table 4:")[
  Median CNV size counts for all consensus-based call sets across coverages with different caller agreement requirements.
  Counts are color mapped with higher values in green and lower values in yellow.
  Call sets only include CNVs that fall within the analysis window of \[500 bp, 1 Mb\].
]

#v(0.8em)

== Statistical Distribution Figures

#cap("Figure 6:")[
  Probability density graph for the precision, recall, and f1-scores of CNVs identified from all tested CNV calling methodologies against the merged benchmark CNV call set derived from 1000G phase 3, HGSVC3, and ONT Vienna.
  Solid lines represent distributions including both Deletions and Duplications; dashed lines represent distributions only including Deletions, and dotted lines represent distributions only including Duplications.
  100 log-spaced interval points between 500bp and 1 Mb were used to estimate the probability distribution shown, and the values of the points are visualized as hollow points scattered throughout the graph with colors corresponding to the respective tested call set.
  Individual points are only shown for the distribution lines including Deletion and Duplication records; the points for the individual SV type distributions were computed separately but omitted from the graph for clarity.
  The distribution curves shown were derived from Gaussian kernel smoothing (σ=5.0) of the points.
]

== Post-Processing And Removal of Artifactual CNV Calls

#figplaceholder("image20 — Figure 2, percent change in calls after excluding problematic regions")

#cap("Figure 2:")[
  Percent change in the amount of CNV calls from each of the three CNV calling tools tested (CNVpytor, Delly, GATK-gCNV) after excluding calls that had at least 50% of their length contained within problematic regions.
  The percent change is in reference to the original number of calls output by each caller.
  This was performed separately for each coverage type and for Deletion (top row) and Duplication (bottom row) CNV records.
  Problematic regions considered for exclusion included centromeres, telomeres, heterochromatin, short-arms, decoy contigs, and alternative contigs.
]

#v(0.8em)

After CNV calling across all coverage types, we applied a post-processing filter to remove CNV intervals overlapping genomic regions that had been excluded during caller preparation (centromeres, telomeres, short arms, heterochromatin, decoy/alternate contigs).
This step succeeded in ensuring that misinterpretation of the lack of reads in these genomic regions did not propagate into false CNV calls by the tools used.
Each caller demonstrates a different behavior in generating these erroneous calls, which is illustrated in Figure 2.
The identification and removal of artifactual calls in this step ensured more reliable downstream results.

== Sequencing-based call sets yield more CNVs than the SNP-array call set, with strong size-distribution differences from the benchmark

To assess the quality of the Sequencing-based call sets, we compared them to the benchmark set and SNP-array call set.

benchmark set and SNP-array call set description

We therefore focused downstream analyses on the 2/3 consensus call sets as a pragmatic middle ground: support by at least two independent callers reduces the likelihood that retained CNVs reflect characteristic behavior of any single algorithm, while preserving substantially more signal than a 3/3 policy.
This concordance principle is commonly used in SV/CNV integration workflows to balance sensitivity and precision under algorithmic limitations along with realistic breakpoint and representation variability #c[Ho 2020] #c[Liu 2022].

== 2/3 consensus call set comparison/evaluation

#figplaceholder("image21 — Figure 4, distribution of caller sources in 2/3 consensus calls")

#cap("Figure 4:")[
  Distribution of the sources from all CNV calls that had consensus between at least 2 of the 3 CNV calling tools used (CNVpytor, Delly, and GATK-gCNV).
  Distributions were generated using the 13 input samples from sequencing data of differing coverage (30x, 6x, 4x, 2x).
  Analysis was performed separately for Deletion (top row) and Duplication (bottom row) CNV records, and separately for the raw caller incidence (left column) and caller combination incidence (right column).
]

#v(0.8em)

For most call categories, caller contributions to consensus CNVs were broadly similar, with overlapping interquartile ranges across many categories.
Higher coverage consensus calls (30x and 6x) yielded significantly higher contributions from GATK-gCNV, and exhibited greater agreement between all three callers.
Conversely, 2x and 4x coverage had much less contribution from GATK-gCNV and instead attributed a more significant percentage of consensus calls from agreement between only CNVpytor and Delly.

Across coverage types, duplication calls exhibited fewer average calls per sample and greater variability in caller contribution than the deletion calls.
The medians of the duplication records are more similar across coverage types than the deletion calls, and additionally do not show any clear trends between coverage degree and caller results.

To capture different aspects of classification performance, we report F-scores across beta values of ½, 1, and 2, which place progressively greater weight on recall relative to precision.
The F₁/₂ score is most directly relevant for cytogenetic applications, where false positives carry higher operational cost than false negatives; however, because the relative performance rankings of the evaluated call sets remained consistent across all three beta values, the choice of F-score does not materially alter the conclusions drawn from the data.

= To-do

- #box(width: 0.9em, height: 0.9em, stroke: 0.6pt) Fix legend ordering of all graphs in main section and supplementals
- #box(width: 0.9em, height: 0.9em, stroke: 0.6pt) Fix titles for all graphs in main section and supplementals
