from cyvcf2 import VCF
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
import pandas as pd


class BenchmarkParser:
    """
    Parse benchmark VCF files and extract per-sample CNV calls.
    Handles multiple benchmark formats with different END position and SVTYPE schemes.
    """
    
    def __init__(self, benchmark_map: Dict[str, str]):
        """
        Initialize BenchmarkMerger with benchmark file paths.
        
        Args:
            benchmark_map: Dictionary mapping benchmark names to VCF file paths
                          Example: {'1000G': '/path/to/1000G.vcf.gz', ...}
        """
        self.benchmark_map = benchmark_map
    
    def sanitize_svtype(self, svtype: Optional[str], record_id: str = "") -> str:
        """
        Sanitize SVTYPE to unified DEL/DUP classification.
        Maps all insertion types (INS, LINE1, ALU, SVA) to DUP for consistency.
        
        Args:
            svtype: SVTYPE from INFO field
            record_id: Record ID (may contain type information for CNV records)
            
        Returns:
            Sanitized type: 'DEL', 'DUP', or 'NA'
        """
        if svtype is None:
            return 'NA'
        
        svtype_upper = svtype.upper()
        
        # Handle deletions - check if any deletion pattern is in the string
        if any(pattern in svtype_upper for pattern in {'DEL', 'DELETION'}):
            return 'DEL'
        
        # Handle duplications and insertions - check if any dup/ins pattern is in the string
        if any(pattern in svtype_upper for pattern in {'DUP', 'DUPLICATION', 'INS', 'INSERTION'}):
            return 'DUP'
        
        # Handle mobile element insertions as duplications
        if any(pattern in svtype_upper for pattern in {'LINE1', 'ALU', 'SVA'}):
            return 'DUP'
        
        # Handle CNV type by checking ID field
        if 'CNV' in svtype_upper:
            record_id_upper = record_id.upper()
            if 'DEL' in record_id_upper:
                return 'DEL'
            elif 'DUP' in record_id_upper:
                return 'DUP'
        
        # Explicitly handle known SVTYPEs that we want to ignore (e.g., BND, INV)
        if any(pattern in svtype_upper for pattern in {'BND', 'INV', 'TRA', 'CTX'}):
            return 'NA'
        
        print(f"Warning: Unrecognized SVTYPE '{svtype}' in record ID '{record_id}'. Skipping.")
        exit(1)
        return 'NA'
    
    def ensure_chr_prefix(self, chrom: str) -> str:
        """Ensure chromosome name has 'chr' prefix."""
        if not chrom.startswith('chr'):
            return f'chr{chrom}'
        return chrom
    
    def parse_benchmark_vcf(
        self, 
        vcf_path: str, 
        sample_ids: Optional[List[str]] = None,
        genome_file_path: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Parse benchmark VCF file and extract CNV calls for specified samples.
        
        Args:
            vcf_path: Path to VCF/BCF file
            sample_ids: List of sample IDs to extract (None = all samples)
            
        Returns:
            Dictionary mapping sample_id -> DataFrame with columns:
            [chrom, start, end, id, svtype, genotype]
        """
        print(f"Processing {vcf_path}")
        
        # Get valid chromosomes from genome file if provided
        valid_chroms = set()
        if genome_file_path is not None:
            try:
                with open(genome_file_path) as f:
                    for line in f:
                        chrom = line.split()[0]
                        valid_chroms.add(self.ensure_chr_prefix(chrom))
                print(f"Loaded {len(valid_chroms)} valid chromosomes from {genome_file_path}")
            except Exception as e:
                print(f"Error reading genome file {genome_file_path}: {e}")
                print("Proceeding without chromosome filtering.")
        else:
            print("No genome file provided, skipping chromosome filtering.")


        # Parse VCF
        sample_data = {}
        
        try:
            vcf = VCF(vcf_path)
            
            # Get samples to process
            if sample_ids is None:
                samples_to_process = vcf.samples
            else:
                samples_to_process = [s for s in sample_ids if s in vcf.samples]
                if not samples_to_process:
                    print(f"Warning: None of the requested samples found in {vcf_path}")
                    return {}
            
            # Initialize data storage for each sample
            for sample in samples_to_process:
                sample_data[sample] = []
            
            # Get sample indices
            sample_indices = {sample: vcf.samples.index(sample) for sample in samples_to_process}
            
            # Process records
            for record in vcf:
                # Skip if chromosome is not in valid set (if genome file provided)
                chrom = self.ensure_chr_prefix(record.CHROM)
                if valid_chroms and chrom not in valid_chroms:
                    continue

                # Skip records without ALT alleles
                if not record.ALT or len(record.ALT) == 0:
                    continue
                
                # Extract basic info
                start = record.POS - 1  # Convert to 0-based
                record_id = record.ID if record.ID else "."
                
                # Extract END - try INFO field first, then calculate from SVLEN
                end = record.INFO.get('END')
                if end is not None:
                    end = int(end)
                else:
                    svlen = record.INFO.get('SVLEN')
                    if svlen is not None:
                        end = record.POS + abs(int(svlen))
                
                # Skip if we couldn't determine END
                if end is None:
                    continue

                # Extract and sanitize SVTYPE
                raw_svtype = record.INFO.get('SVTYPE')
                svtype = self.sanitize_svtype(raw_svtype, record_id)
                
                # Skip if SVTYPE is not DEL or DUP
                if svtype == 'NA':
                    continue
                
                # Extract genotypes for requested samples
                genotypes = record.genotypes
                
                for sample, sample_idx in sample_indices.items():
                    # Get genotype for this sample
                    gt = genotypes[sample_idx][:2]  # First two elements are the alleles
                    
                    # Filter: keep only non-zero, non-missing genotypes
                    # gt is like [0, 0], [0, 1], [1, 1], [-1, -1] (missing), etc.
                    if any(allele != 0 for allele in gt):
                        # Convert genotype to string (e.g., "0/1", "1/1")
                        gt_str = f"{gt[0]}/{gt[1]}"
                        
                        sample_data[sample].append({
                            'chrom': chrom,
                            'start': start,
                            'end': end,
                            'id': record_id,
                            'svtype': svtype,
                            'genotype': gt_str
                        })
            
        except Exception as e:
            print(f"Error processing VCF file {vcf_path}: {e}")
            import traceback
            traceback.print_exc()
        
        # Convert to DataFrames
        result = {}
        for sample, records in sample_data.items():
            if records:
                result[sample] = pd.DataFrame(records)
            else:
                print(f"Warning: No variants found for sample {sample}")
                result[sample] = pd.DataFrame(columns=['chrom', 'start', 'end', 'id', 'svtype', 'genotype'])
        
        return result
    
    def write_sample_beds(
        self, 
        sample_df: pd.DataFrame, 
        output_dir: str, 
        sample_id: str
    ) -> Dict[str, int]:
        """
        Write per-sample BED files split by SV type.
        
        Args:
            sample_df: DataFrame with sample variants
            output_dir: Directory to write BED files
            sample_id: Sample identifier
            
        Returns:
            Dictionary with counts: {'all': N, 'DEL': N, 'DUP': N}
        """
        # Create sample-specific directory
        sample_dir = Path(output_dir) / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        counts = {'all': 0, 'DEL': 0, 'DUP': 0}
        
        if sample_df.empty:
            return counts
        
        counts['all'] = len(sample_df)
        
        # Prepare output (drop genotype column for BED format)
        output_df = sample_df[['chrom', 'start', 'end', 'id', 'svtype', 'genotype']]
        
        # Write all variants
        all_bed = sample_dir / f"{sample_id}.bed"
        output_df.to_csv(all_bed, sep='\t', header=False, index=False)
        
        # Write DEL variants
        del_df = sample_df[sample_df['svtype'] == 'DEL']
        counts['DEL'] = len(del_df)
        if not del_df.empty:
            del_bed = sample_dir / f"{sample_id}.DEL.bed"
            del_df[['chrom', 'start', 'end', 'id', 'svtype', 'genotype']].to_csv(
                del_bed, sep='\t', header=False, index=False
            )
        
        # Write DUP variants
        dup_df = sample_df[sample_df['svtype'] == 'DUP']
        counts['DUP'] = len(dup_df)
        if not dup_df.empty:
            dup_bed = sample_dir / f"{sample_id}.DUP.bed"
            dup_df[['chrom', 'start', 'end', 'id', 'svtype', 'genotype']].to_csv(
                dup_bed, sep='\t', header=False, index=False
            )
        
        return counts
    
    def parse_all_benchmarks_to_bed(
        self, 
        output_base_dir: str | Path, 
        sample_ids: Optional[List[str]] = None,
        common_samples_only: bool = False,
        genome_file_path: Optional[str] = None
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Process all benchmarks and write per-sample BED files.
        
        Args:
            output_base_dir: Base directory for all outputs
            sample_ids: List of sample IDs to extract (None = all samples)
            
        Returns:
            Nested dictionary: benchmark -> sample -> counts
        """
        
        # If common_samples_only is True, we will first determine the set of samples present in all benchmarks
        if common_samples_only:
            sample_sets = []
            for benchmark_name, vcf_path in self.benchmark_map.items():
                try:
                    vcf = VCF(vcf_path)
                    sample_sets.append(set(vcf.samples))
                except Exception as e:
                    print(f"Error reading samples from {vcf_path}: {e}")
                    sample_sets.append(set())
            common_samples = set.intersection(*sample_sets) if sample_sets else set()
            print(f"Common samples across all benchmarks: {len(common_samples)}")
            sample_ids = list(common_samples)
        
        
        all_results = {}
        
        for benchmark_name, vcf_path in self.benchmark_map.items():
            print(f"\n{'='*60}")
            print(f"Processing benchmark: {benchmark_name}")
            print(f"{'='*60}")
            
            if not os.path.exists(vcf_path):
                print(f"Warning: File not found: {vcf_path}")
                continue
            
            # Parse VCF
            sample_data = self.parse_benchmark_vcf(vcf_path, sample_ids, genome_file_path=genome_file_path)
            
            # Create output directory for this benchmark
            output_dir = Path(output_base_dir) / benchmark_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write BED files for each sample
            benchmark_results = {}
            for sample_id, sample_df in sample_data.items():
                counts = self.write_sample_beds(sample_df, str(output_dir), sample_id)
                benchmark_results[sample_id] = counts
                
                print(f"Sample {sample_id}: {counts['all']} total, "
                      f"{counts['DEL']} DEL, {counts['DUP']} DUP")
                
                # Sanity check
                if counts['all'] != (counts['DEL'] + counts['DUP']):
                    print(f"  Warning: Counts do not match! "
                          f"Total: {counts['all']}, DEL: {counts['DEL']}, DUP: {counts['DUP']}")
            
            all_results[benchmark_name] = benchmark_results
            print(f"\nCompleted {benchmark_name}: {len(sample_data)} samples processed")
        
        return all_results
    
    def merge_across_benchmarks(
        self,
        output_base_dir: str | Path,
        merge_distance: int = 0,  # bp to merge nearby intervals,
        genome_file_path: Optional[str] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Merge BED files across all benchmarks for each sample+svtype.
        
        Args:
            output_base_dir: Base directory containing benchmark subdirectories
            merge_distance: Distance for merging nearby intervals (default: 0)
            
        Returns:
            Dictionary: sample -> {'DEL': count, 'DUP': count}
        """
        import subprocess
        from collections import defaultdict
        
        base_path = Path(output_base_dir)
        merged_dir = base_path / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all samples across benchmarks
        samples_svtypes = defaultdict(lambda: defaultdict(list))
        
        for benchmark_name in self.benchmark_map.keys():
            benchmark_dir = base_path / benchmark_name
            if not benchmark_dir.exists():
                continue
                
            for sample_dir in benchmark_dir.iterdir():
                if not sample_dir.is_dir():
                    continue
                sample_id = sample_dir.name
                
                # Collect DEL and DUP files
                for svtype in ['DEL', 'DUP']:
                    bed_file = sample_dir / f"{sample_id}.{svtype}.bed"
                    if bed_file.exists():
                        samples_svtypes[sample_id][svtype].append(bed_file)
        
        # Merge for each sample+svtype
        results = {}
        for sample_id, svtype_files in samples_svtypes.items():
            sample_merged_dir = merged_dir / sample_id
            sample_merged_dir.mkdir(exist_ok=True)
            results[sample_id] = {}
            
            for svtype, bed_files in svtype_files.items():
                if not bed_files:
                    continue
                    
                output_file = sample_merged_dir / f"{sample_id}.merged.{svtype}.bed"
                
                if genome_file_path is not None:
                    sort_command = f"bedtools sort -g {genome_file_path} -i -"
                else:
                    sort_command = "sort -k1,1 -k2,2n"
    
                # Concatenate, sort, and merge
                # Build command as string for shell execution (pipes and redirects require shell=True)
                bed_files_str = ' '.join(str(bf) for bf in bed_files)
                command = f"cat {bed_files_str} | {sort_command} | bedtools merge -i - > {output_file}"
                
                subprocess.run(command, shell=True, check=True)
                
                # Count results
                with open(output_file) as f:
                    count = sum(1 for _ in f)
                results[sample_id][svtype] = count
                
                print(f"{sample_id} {svtype}: {len(bed_files)} files -> {count} merged intervals")
        
        return results

