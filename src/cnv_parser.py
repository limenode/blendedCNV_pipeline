from cyvcf2 import VCF
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import glob
import pandas as pd

class CNVParser:
    def __init__(self, input_map_file: str):
        self.tool_patterns = {}
        self.parse_input_map(input_map_file)

    def parse_input_map(self, input_map_file: str):
        """Parse the input map file to extract tool names and path patterns."""
        with open(input_map_file, 'r') as f:
            for line in f:
                if line and line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    print (f"Warning: Skipping malformed line: {line.strip()}")
                    continue
                tool_name, path_pattern = parts[0], parts[1]
                self.tool_patterns[tool_name] = path_pattern
    
    def has_id_field(self, pattern: str) -> bool:
        """Check if the path pattern contains an {id} field."""
        return '{id}' in pattern
    
    def get_sample_id_from_vcf(self, vcf_path: str) -> str:
        """Extract sample ID from the VCF file."""
        try:
            vcf = VCF(vcf_path)
            if vcf.samples:
                return vcf.samples[0]
            else:
                print(f"Warning: No samples found in VCF file {vcf_path}")
                return Path(vcf_path).stem
        except Exception as e:
            print(f"Error reading VCF file {vcf_path}: {e}")
            return Path(vcf_path).stem
        
    def extract_id_from_pattern(self, path: str, pattern: str) -> str:
        """Extract sample ID from the file path based on the pattern."""

        # Convert pattern to regex
        regex_pattern = pattern.replace('{id}', r'([^/]+)')
        regex_pattern = regex_pattern.replace('*', r'[^/]*')

        match = re.search(regex_pattern, path)
        if match:
            return match.group(1)
        else:
            print(f"Warning: Could not extract ID from path {path} using pattern {pattern}")
            return Path(path).stem
        
    def find_vcf_files(self, tool_name: str, base_dir: str = ".") -> List[Tuple[str, str]]:
        """Find VCF files for a given tool based on its path pattern."""
        if tool_name not in self.tool_patterns:
            print(f"Warning: Tool {tool_name} not found in input map.")
            return []
        
        pattern = self.tool_patterns[tool_name]
        search_pattern = pattern.replace('{id}', '*')
        full_search_pattern = os.path.join(base_dir, search_pattern)
        
        vcf_files = glob.glob(full_search_pattern, recursive=True)
        results = []
        
        for vcf_path in vcf_files:
            if self.has_id_field(pattern):
                sample_id = self.extract_id_from_pattern(vcf_path, pattern)
            else:
                sample_id = self.get_sample_id_from_vcf(vcf_path)
            results.append((sample_id, vcf_path))
        
        return results
    
    def get_all_vcf_files(self, base_dir: str = ".") -> Dict[str, List[Tuple[str, str]]]:
        """Get all VCF files for all tools."""
        all_results = {}
        for tool_name in self.tool_patterns.keys():
            all_results[tool_name] = self.find_vcf_files(tool_name, base_dir)
        return all_results


    def determine_sex_threshold(self, vcf_path: str) -> float:
        """
        Determine sex chromosome threshold (1.0 for XY, 2.0 for XX) based on sex chromosome RDCN values.
        
        Args:
            vcf_path: Path to VCF file
            
        Returns:
            Threshold value (1.0 or 2.0)
        """
        sex_chrom_rdcn = []
        try:
            vcf = VCF(vcf_path)
            for record in vcf:
                if record.CHROM in ['chrX', 'chrY', 'X', 'Y']:
                    # Get RDCN from FORMAT field of first sample
                    rdcn = record.format('RDCN')
                    if rdcn is not None and len(rdcn) > 0:
                        sex_chrom_rdcn.append(float(rdcn[0][0]))
        except Exception as e:
            print(f"Error determining sex threshold for {vcf_path}: {e}")
            return 2.0  # Default to XX
        
        if not sex_chrom_rdcn:
            return 2.0  # Default to XX if no sex chromosome data
        
        # Calculate median RDCN for sex chromosomes
        median_rdcn = sorted(sex_chrom_rdcn)[len(sex_chrom_rdcn) // 2]
        
        # If median is closer to 1.0, likely XY; if closer to 2.0, likely XX
        return 1.0 if median_rdcn < 1.5 else 2.0
    
    def sanitize_svtype(self, svtype: str) -> str:
        """Sanitize SVTYPE to DEL, DUP, or NA."""

        if svtype is None:
            return 'NA'
        svtype = svtype.upper()
        if svtype in {'DEL', 'DELETION'}:
            return 'DEL'
        elif svtype in {'DUP', 'DUPLICATION', 'INS', 'INSERTION'}:
            return 'DUP'
        else:
            return 'NA'
    
    def extract_svtype_from_info(self, record) -> str:
        """Extract SVTYPE from INFO field."""
        svtype = record.INFO.get('SVTYPE')
        return self.sanitize_svtype(svtype)
    
    def extract_svtype_from_alt(self, record) -> str:
        """Extract SVTYPE from ALT field."""
        if record.ALT and len(record.ALT) > 0:
            alt_str = str(record.ALT[0]).strip()
            if alt_str.startswith('<') and alt_str.endswith('>'):
                return self.sanitize_svtype(alt_str[1:-1])  # Remove angle brackets
            return self.sanitize_svtype(alt_str)
        return 'NA'
    
    def extract_svtype_from_rdcn(self, record, sex_threshold: float) -> str:
        """Extract SVTYPE from RDCN FORMAT field."""
        # Get RDCN from FORMAT field of first sample
        rdcn = record.format('RDCN')
        if rdcn is None or len(rdcn) == 0:
            return 'NA'
        
        rdcn = float(rdcn[0][0])
        chrom = record.CHROM
        
        if chrom in ['chrX', 'chrY', 'X', 'Y']:
            threshold = sex_threshold
        else:
            threshold = 2.0  # Autosomal threshold
        
        return 'DUP' if rdcn > threshold else 'DEL'
    
    def determine_svtype_method(self, vcf_path: str) -> Callable:
        """
        Determine which method to use for SVTYPE extraction by examining the first record.
        Returns the appropriate extraction function.
        
        Args:
            vcf_path: Path to VCF file
            
        Returns:
            Function to extract SVTYPE from a record
        """
        try:
            vcf = VCF(vcf_path)
            for record in vcf:
                # Check INFO SVTYPE
                if record.INFO.get('SVTYPE') is not None:
                    if self.sanitize_svtype(record.INFO.get('SVTYPE')) != 'NA':
                        return self.extract_svtype_from_info
                
                # Check ALT field
                if not record.ALT:
                    continue

                if len(record.ALT) > 0:
                    alt_str = str(record.ALT[0]).strip()
                    if alt_str.startswith('<') and alt_str.endswith('>'):
                        if self.sanitize_svtype(alt_str[1:-1]) != 'NA':
                            return self.extract_svtype_from_alt
                
                # Check RDCN in FORMAT field
                rdcn = record.format('RDCN')
                if rdcn is not None and len(rdcn) > 0:
                    sex_threshold = self.determine_sex_threshold(vcf_path)
                    return lambda rec: self.extract_svtype_from_rdcn(rec, sex_threshold)
                
                # Only check first record
                break

        except Exception as e:
            print(f"Error determining SVTYPE method for {vcf_path}: {e}")
        
        print("Defaulting to SVTYPE from INFO field.")
        return self.extract_svtype_from_info  # Default to info method

    def extract_end_position(self, record) -> int:
        """Extract END position from VCF record."""
        end = record.INFO.get('END')
        if end is not None:
            return int(end)
        return record.POS # If END is not available, use POS as a fallback

    def convert_vcf_to_bed(self, vcf_path: str, source: str = "") -> pd.DataFrame:
        """
        Convert VCF file to BED format DataFrame.
        
        Args:
            vcf_path: Path to VCF/BCF file
            
        Returns:
            DataFrame with columns: chrom, start, end, svtype
        """
        # Determine extraction function to use
        svtype_extractor = self.determine_svtype_method(vcf_path)
        
        records = []
        try:
            vcf = VCF(vcf_path)
            for record in vcf:
                # Exclude empty ALT alleles
                if not record.ALT or len(record.ALT) == 0:
                    continue


                chrom = record.CHROM
                start = record.POS - 1  # Convert to 0-based
                end = self.extract_end_position(record)
                svtype = svtype_extractor(record)
                
                records.append((chrom, start, end, svtype))

        except Exception as e:
            print(f"Error processing VCF file {vcf_path}: {e}")
            
        df = pd.DataFrame(records, columns=['chrom', 'start', 'end', 'svtype'])
        if source:
            df['source'] = source
        return df
    