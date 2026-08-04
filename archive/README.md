# archive/

Retired code, kept for reference while the computation phase is rewritten on the
`consensuscnv.callsets` layer. **Nothing here is imported by the live pipeline**,
and it is deliberately outside `src/` so it cannot be.

Everything was moved (via `git mv`, so `git log --follow` still works) from the
tree as of commit `70e52b4`.

## What's here and why

| path | was | why archived |
|---|---|---|
| `computation/` | `src/consensuscnv/computation/` | The whole old computation phase: `consensus_calling.py` (networkx consensus), `computation_functions.py` (node-level binary classification), `computation_driver.py`. Superseded by `consensuscnv.callsets` for merging; the classification half is the next thing to be rewritten, so keep this readable until then. |
| `overlap_graph.py` | `src/consensuscnv/overlap_graph.py` | The networkx `Graph` representation. Replaced by `CallSet` — flat numpy edge lists, sorted by key, sliced per threshold. |
| `calls.py` | `src/consensuscnv/calls.py` | The old `Call` (with `sources: frozenset`, `membership`, `parent_calls`). Replaced by `consensuscnv.callsets.calls.Call`, which is `(chrom, start, end, svtype, source, sample_id)` — sources are an interned bitmask on the `CallSet` now, and `membership` is gone. |
| `benchmark_distances.py` | a function in `analysis/analysis_functions.py` | `discover_distances_between_benchmark_cnvs` was the only consumer of `calls` / `overlap_graph` outside `computation/`, and had no callers itself. Archived so those two could be retired. Its output is arguably already available as `CallSet.gap_key`. |
| `test_computation.py` | `src/test_computation.py` | marimo notebook driving the archived computation code. |

## Things worth knowing before reviving any of it

- **The classification → analysis link was broken in two independent ways**, so
  the analysis phase never read any of this. The writer emits
  `{sample}.{label}.bed` (2 parts) while `load_analysis_data.discover_classification_files`
  requires 3-part stems; and the writer used the 4-level parameterized tree while
  `analysis_driver.py` looked in the legacy single-level `classification_root()`.
  Fix both when wiring up the replacement.
- **There was never a parquet classification store.** An earlier `CLAUDE.md`
  described a hive-partitioned `computation/classification_store.py` with
  `read_classification` / `export_bed_tree`. No such file exists in this
  repository or anywhere in its git history. Classification wrote plain BEDs.
- `get_binary_classification.sh` is referenced by `src/consensuscnv.egg-info/SOURCES.txt`
  but was already deleted before this migration.

## Still live, deliberately not archived

`analysis/load_analysis_data.py` looks dead but is not — its `filter_by_size` is
imported by both `analysis_functions.py` and `cnv_plotter.py`. Only its
`discover_classification_files` / `build_analysis_data_structure` / `load_*_file`
half is dead, and that half should be *rewritten* against the new classification
output rather than archived.
