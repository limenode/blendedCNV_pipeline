import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## imports
    """)
    return


@app.cell
def _():
    import glob
    from line_profiler import LineProfiler
    import pandas as pd
    import os
    from pathlib import Path

    return Path, glob, os


@app.cell
def _():
    import consensuscnv
    import consensuscnv.utils
    from consensuscnv import overlap_graph

    return (consensuscnv,)


@app.cell
def _():
    from consensuscnv import (
        computation,
        parsing,
    )

    return computation, parsing


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## prep
    """)
    return


@app.cell
def _(Path, consensuscnv, os, parsing):
    config = consensuscnv.utils.build_config(
            config_path=Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/internal.config.yaml"),
            do_processing=True,
            do_analysis=True,
            do_computation=True,
        )
    layout = config.layout

    # Check path for if parsing is needed 
    path_to_test_parsing = "/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/4x_Coverage/cnvpytor"
    if os.path.exists(path_to_test_parsing):
        print("Parsing is not needed, files already exist.")
    else:
        print("Parsing is needed, running parsing function.")
        # Run parsing function
        parsing.parse_input_files(config=config)
    return config, layout


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Testing binary classification
    """)
    return


@app.cell
def _():
    from consensuscnv.output_layout import (
        BenchmarkMergeParams,
        ClassificationParams,
        ConsensusParams,
    )
    from consensuscnv.computation.consensus_calling import (
        load_benchmark_graph,
        merge_benchmarks,
    )
    from consensuscnv.overlap_graph import dump_calls_to_bed

    return (
        BenchmarkMergeParams,
        ClassificationParams,
        ConsensusParams,
        dump_calls_to_bed,
        load_benchmark_graph,
        merge_benchmarks,
    )


@app.cell
def _(ConsensusParams, computation, config):
    # Sweep consensus tunables here (one call emits all three levels each time).
    consensus_param_sweep = [ConsensusParams(min_weight=w) for w in [0.0, 0.5]]

    consensus_out = computation.compute_experimental_consensus(
        config,
        weights=[0.0, 0.5],
    )
    return (consensus_out,)


@app.cell
def _(config, consensus_out, dump_calls_to_bed, layout):
    for _key, _calls in consensus_out.items():
        _experimental_key, _weight, _level = _key

        output_dir = layout.consensus_rep_dir(
            origin_set=_experimental_key,
            level=_level,
            representation=f"weight_{_weight}"
        )

        dump_calls_to_bed(_calls, output_dir, chrom_order=config.chromosome_order)
    return


@app.cell
def _(BenchmarkMergeParams, config, load_benchmark_graph):
    # Build the benchmark graph once, reuse across padding values.
    benchmark_graph = load_benchmark_graph(config)

    benchmark_param_sweep = [
        BenchmarkMergeParams(padding=p, min_weight=0.0, link_same_source=True)
        for p in [0, 500, 1000, 2000]
    ]
    return benchmark_graph, benchmark_param_sweep


@app.cell
def _(ClassificationParams):
    classification_param_sweep = [
        ClassificationParams(reciprocal_threshold=t) for t in [0.0, 0.3, 0.5, 0.7]
    ]
    return (classification_param_sweep,)


@app.cell
def _(glob, layout):
    _all_consensus_beds = glob.glob(str(layout.root / "*" / "consensus_*" / "*" / "*.bed"))
    _all_consensus_beds
    return


@app.cell
def _(Path, config, glob, layout):
    def build_io_sets(benchmark_params, classification_params, benchmark_bed_path):
        """Every tested call set -> its classification output dir for one setting."""

        def out(input_set_key, call_set):
            return str(layout.classification_dir(
                input_set_key,
                call_set,
                benchmark_params=benchmark_params,
                classification_params=classification_params,
            ))

        # Format: (input_set_dir, output_dir, benchmark_bed_path)
        io_sets: list[tuple[str, str, str]] = []

        # Consensus call sets
    
        _all_consensus_bed_dirs = glob.glob(str(layout.root / "*" / "consensus_*" / "*"))
        for _path in _all_consensus_bed_dirs:
            _split = Path(_path).parts 
            _experimental_key = _split[-3]
            _level = int(_split[-2].split("_")[-1])
            _weight = _split[-1]

            io_sets.append((
                str(layout.consensus_rep_dir(_experimental_key, _level, representation=_weight)),
                out(_experimental_key, f"consensus_weight_{_weight}_{_level}"),
                str(benchmark_bed_path),
            ))    

        # Individual callers.
        for exp_key, tools in config.experimental.items():
            for tool in tools:
                io_sets.append((str(layout.bed_tool_dir(exp_key, tool)), out(exp_key, tool), str(benchmark_bed_path)))

        # Controls.
        for ctrl_key in config.control:
            io_sets.append((str(layout.control_bed_dir(ctrl_key)), out(ctrl_key, "calls"), str(benchmark_bed_path)))

        return io_sets

    return (build_io_sets,)


@app.cell
def _(
    benchmark_graph,
    benchmark_param_sweep,
    build_io_sets,
    classification_param_sweep,
    computation,
    config,
    merge_benchmarks,
):
    # Full sweep: (benchmark padding) x (reciprocal threshold), consensus params folded
    # into the tested call sets. Benchmark merge reuses the prebuilt graph.
    for benchmark_params in benchmark_param_sweep:
        benchmark_bed_path = merge_benchmarks(config, benchmark_params, benchmark_graph)
        for classification_params in classification_param_sweep:
            io_sets = build_io_sets(benchmark_params, classification_params, benchmark_bed_path)
            print(f"Classifying {len(io_sets)} sets @ {benchmark_params.slug()} / {classification_params.slug()}")
            computation.run_binary_classification_script(
                config,
                io_sets,
                reciprocal_threshold=classification_params.reciprocal_threshold,
            )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
