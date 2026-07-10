[] Switch to a single-file per sample instead of separating by DEL and DUP
- Improves consistency and the file system becomes much more managable.

[] Check merge_components in overlap_graph.py
- Need to refine limiter to be on the unique callers and not number of nodes
- Should switch from "MergedIntervals" to "Call"

[] Only move get_binary_classification into graph operation if needed.