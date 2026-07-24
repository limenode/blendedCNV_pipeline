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
    import os
    import pandas as pd
    from pathlib import Path

    return LineProfiler, Path, glob, os


@app.cell
def _():
    import consensuscnv
    import consensuscnv.utils
    import consensuscnv.overlap_graph as overlap_graph

    return (consensuscnv,)


@app.cell
def _():
    from consensuscnv import (
        analysis,
        computation,
        parsing,
    )

    return computation, parsing


@app.cell
def _():
    from consensuscnv.overlap_graph import (
        generate_graph_from_calls,
        merge_component,
        read_bed_file,
        resolve_components,
        resolve_graph,
    )

    return (
        generate_graph_from_calls,
        read_bed_file,
        resolve_components,
        resolve_graph,
    )


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
    path_to_test_parsing = "/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/4x_Coverage/bed/cnvpytor"
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
    ## Testing subgraph view retrieval
    """)
    return


@app.cell
def _(Path, config, glob, layout, read_bed_file):
    benchmark_keys = config.benchmark.keys()

    benchmark_calls1 = []
    for bm_key in benchmark_keys:
        bed_paths = glob.glob(str(layout.benchmark_dir(bm_key)) + "/*.bed")
        for path in bed_paths:
            if Path(path).is_file():
                benchmark_calls1.extend(read_bed_file(Path(path), membership=bm_key))
    print(f"Total calls read: {len(benchmark_calls1)}")
    return (benchmark_calls1,)


@app.cell
def _(benchmark_calls1, generate_graph_from_calls):
    benchmark_graph1 = generate_graph_from_calls(benchmark_calls1)
    return (benchmark_graph1,)


@app.cell
def _(LineProfiler, benchmark_graph1, resolve_graph):
    lp1 = LineProfiler()
    lp1.add_function(resolve_graph)
    lp1.enable_by_count()
    test_resolve_graph_p0 = resolve_graph(
        benchmark_graph1,
        min_nodes=1,
        min_weight=0.0,
        padding=0,
        link_same_source=True
    )
    lp1.disable_by_count()
    lp1.print_stats()

    print(f"Original Graph Nodes: {len(benchmark_graph1.nodes)}, Edges: {len(benchmark_graph1.edges)}")
    print(f"Resolved Graph (Padding=0) Nodes: {len(test_resolve_graph_p0.nodes)}, Edges: {len(test_resolve_graph_p0.edges)}")
    return


@app.cell
def _(LineProfiler, benchmark_graph1, resolve_components):
    lp2 = LineProfiler()
    lp2.add_function(resolve_components)
    lp2.enable_by_count()
    test_resolve_components_p0 = resolve_components(
        benchmark_graph1,
        min_nodes=1,
        min_weight=0.0,
        padding=0,
        link_same_source=True
    )
    lp2.disable_by_count()
    lp2.print_stats()
    return


@app.cell
def _(benchmark_graph1, resolve_components):
    benchmark_components_p0 = resolve_components(
        benchmark_graph1, padding=0, link_same_source=True
    )
    benchmark_components_p1000 = resolve_components(
        benchmark_graph1, padding=1000, link_same_source=True
    )

    print(f"Number of components (Padding=0): {len(benchmark_components_p0)}")
    print(f"Number of components (Padding=1000): {len(benchmark_components_p1000)}")
    return


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

    return (
        BenchmarkMergeParams,
        ClassificationParams,
        ConsensusParams,
        load_benchmark_graph,
        merge_benchmarks,
    )


@app.cell
def _(ConsensusParams, computation, config):
    # Sweep consensus tunables here (one call emits all three levels each time).
    consensus_param_sweep = [ConsensusParams(min_weight=w) for w in [0.5]]

    consensus_bed_paths_by_param = {
        cp: computation.compute_consensus_from_beds(config, cp)
        for cp in consensus_param_sweep
    }
    return (consensus_bed_paths_by_param,)


@app.cell
def _(BenchmarkMergeParams, config, load_benchmark_graph):
    # Build the (threshold-agnostic) benchmark graph once, reuse across padding values.
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
def _(config, consensus_bed_paths_by_param, layout):
    def build_io_sets(benchmark_params, classification_params, benchmark_bed_path):
        """Every tested call set -> its classification output dir for one setting."""

        def out(input_set_key, call_set):
            return str(layout.classification_dir(
                input_set_key,
                call_set,
                benchmark_params=benchmark_params,
                classification_params=classification_params,
            ))

        io_sets: list[tuple[str, str, str]] = []

        # Consensus call sets (keyed by their call-set slug, e.g. consensus_2of3_w0.5).
        for consensus_paths in consensus_bed_paths_by_param.values():
            for input_set, call_set_map in consensus_paths.items():
                for call_set_slug, consensus_bed_path in call_set_map.items():
                    io_sets.append((str(consensus_bed_path), out(input_set, call_set_slug), str(benchmark_bed_path)))

        # Individual callers.
        for exp_key, tools in config.experimental.items():
            for tool in tools:
                io_sets.append((str(layout.bed_tool_dir(exp_key, tool)), out(exp_key, tool), str(benchmark_bed_path)))

        # Controls.
        for ctrl_key in config.control.keys():
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
