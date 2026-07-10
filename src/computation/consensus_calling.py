import glob
from pathlib import Path
import shutil

from utils import PipelineConfig
from overlap_graph import build_sample_graphs_from_beds, merge_components


def compute_consensus_from_beds(config: PipelineConfig, weight_threshold: float = 0.5):
    """Compute consensus calls from per-caller BED files and write them to the output folder."""

    layout = config.layout
    experimental_keys = config.experimental.keys()

    network_paths = {}
    for experimental_key in experimental_keys:
        bed_paths = glob.glob(str(layout.bed_dir(experimental_key)) + "/*/*.bed")
        network_paths[experimental_key] = [
            Path(p) for p in bed_paths if Path(p).is_file()
        ]

    networks = {}

    for experimental_key in experimental_keys:
        networks[experimental_key] = build_sample_graphs_from_beds(
            network_paths[experimental_key]
        )

    for experimental_key, sample_graph_dict in networks.items():
        for sample_id, graph in sample_graph_dict.items():
            for level in (1, 2, 3):
                consensus_calls = merge_components(
                    graph,
                    min_nodes=level,
                    min_weight=weight_threshold,
                    chrom_order=config.chromosome_order,
                )
                output_dir = layout.consensus_rep_dir(experimental_key, level, "unions")
                output_dir.mkdir(parents=True, exist_ok=True)

                # Open the output files for writing
                all_file = open(output_dir / f"{sample_id}.bed", "w")
                del_file = open(output_dir / f"{sample_id}.DEL.bed", "w")
                dup_file = open(output_dir / f"{sample_id}.DUP.bed", "w")
                
                for call in consensus_calls:
                    all_file.write(f"{call.bed_str()}\n")
                    if call.svtype == "DEL":
                        del_file.write(f"{call.bed_str()}\n")
                    elif call.svtype == "DUP":
                        dup_file.write(f"{call.bed_str()}\n")
    
                # Close the output files
                all_file.close()
                del_file.close()
                dup_file.close()

                # Temporarily copy the union files to the "intersection" directory for downstream processing
                intersection_dir = layout.consensus_rep_dir(
                    experimental_key, level, "intersections"
                )
                intersection_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(output_dir / f"{sample_id}.bed", intersection_dir / f"{sample_id}.bed")
                shutil.copy(output_dir / f"{sample_id}.DEL.bed", intersection_dir / f"{sample_id}.DEL.bed")
                shutil.copy(output_dir / f"{sample_id}.DUP.bed", intersection_dir / f"{sample_id}.DUP.bed")
                


def merge_benchmarks(config: PipelineConfig, weight_threshold: float = 0.0):
    """Merge benchmark calls from per-benchmark BED files and write them to the output folder."""

    layout = config.layout
    benchmark_keys = config.benchmark.keys()

    network_paths = []
    for key in benchmark_keys:
        bed_paths = glob.glob(str(layout.benchmark_dir(key)) + "/*.bed")
        network_paths.extend([Path(p) for p in bed_paths if Path(p).is_file()])

    network = build_sample_graphs_from_beds(network_paths)
    
    

    for sample_id, graph in network.items():
        merged_calls = merge_components(
            graph,
            min_nodes=1,
            min_weight=weight_threshold,
            chrom_order=config.chromosome_order,
        )
        output_dir = layout.benchmark_dir("merged")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{sample_id}.bed"

        # Open the output files for writing
        all_file = open(output_file, "w")
        del_file = open(output_dir / f"{sample_id}.DEL.bed", "w")
        dup_file = open(output_dir / f"{sample_id}.DUP.bed", "w")
        
        for call in merged_calls:
            all_file.write(f"{call.bed_str()}\n")
            if call.svtype == "DEL":
                del_file.write(f"{call.bed_str()}\n")
            elif call.svtype == "DUP":
                dup_file.write(f"{call.bed_str()}\n")
        
        # Close the output files
        all_file.close()
        del_file.close()
        dup_file.close()
            
