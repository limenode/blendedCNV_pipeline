import pandas as pd
from dataclasses import dataclass

# BED column layouts emitted by the binary classification step.
INTERVAL_COLUMNS = ["chrom", "start", "end", "svtype", "source"]
TP_COLUMNS = INTERVAL_COLUMNS + [f"truth_{c}" for c in INTERVAL_COLUMNS]

def _read_bed(path, columns):
    """Read a tab-delimited BED into a DataFrame, tolerating empty files."""
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    return pd.read_csv(
        path, sep="\t", header=None, names=columns, dtype={"chrom": str}
    )

def read_interval_bed(path):
    """Read a 5-column interval BED (FP or FN calls)."""
    return _read_bed(path, INTERVAL_COLUMNS)

def read_query_truth_bed(path):
    """Read a 10-column TP BED (query call + its matched truth interval)."""
    return _read_bed(path, TP_COLUMNS)

@dataclass
class SampleClassification:
    """TP/FP/FN call DataFrames for one sample within a single call set."""

    sample_id: str
    tp: pd.DataFrame
    fp: pd.DataFrame
    fn: pd.DataFrame

    @property
    def counts(self):
        """TP/FP/FN row counts as a dict."""
        return {"TP": len(self.tp), "FP": len(self.fp), "FN": len(self.fn)}

def _concat(frames):
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_call_set(call_set_dir):
    """Load per-sample SampleClassification objects from a call-set directory.

    Pools each sample's svtypes (DEL, DUP) into one DataFrame per label.
    """
    samples = sorted({p.name.split(".")[0] for p in call_set_dir.glob("*.bed")})
    result = []
    for sample in samples:
        tp = _concat(
            [read_query_truth_bed(p) for p in sorted(call_set_dir.glob(f"{sample}.*.TP.bed"))]
        )
        fp = _concat(
            [read_interval_bed(p) for p in sorted(call_set_dir.glob(f"{sample}.*.FP.bed"))]
        )
        fn = _concat(
            [read_interval_bed(p) for p in sorted(call_set_dir.glob(f"{sample}.*.FN.bed"))]
        )
        result.append(SampleClassification(sample, tp, fp, fn))
    return result

def _subdirs(path):
    return sorted(p for p in path.iterdir() if p.is_dir())

def load_binary_classification(root):
    """Walk binary_classification/<bench>/<classify>/<set>/<call_set>/ into a nested dict.

    Returns
    {bench_setting: {classify_setting: {input_set: {call_set: [SampleClassification, ...]}}}}.
    """
    tree = {}
    for bench_dir in _subdirs(root):
        classify_map = {}
        for classify_dir in _subdirs(bench_dir):
            set_map = {}
            for set_dir in _subdirs(classify_dir):
                call_set_map = {
                    call_set_dir.name: load_call_set(call_set_dir)
                    for call_set_dir in _subdirs(set_dir)
                }
                set_map[set_dir.name] = call_set_map
            classify_map[classify_dir.name] = set_map
        tree[bench_dir.name] = classify_map
    return tree

def summarize(tree) -> pd.DataFrame:
    """Flatten a loaded tree into a per-sample TP/FP/FN count table with precision/recall."""
    rows = []

    for bench_setting, classify_map in tree.items():
        for classify_setting, set_map in classify_map.items():
            for set_name, call_set_map in set_map.items():
                for call_set, samples in call_set_map.items():
                    for sc in samples:
                        rows.append(
                            {
                                "benchmark_setting": bench_setting,
                                "classification_setting": classify_setting,
                                "input_set": set_name,
                                "call_set": call_set,
                                "sample": sc.sample_id,
                                **sc.counts,
                            }
                        )
    df = pd.DataFrame(rows)
    df["precision"] = df["TP"] / (df["TP"] + df["FP"])
    df["recall"] = df["TP"] / (df["TP"] + df["FN"])
    return df