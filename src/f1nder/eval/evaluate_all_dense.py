from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import pandas as pd
import pyterrier as pt

from f1nder.eval.measures import get_measures
from f1nder.utils.io import ensure_dir, load_queries_json, load_qrels_json, save_metrics_df

# --------------------------------------------------------------------------------------
# Dense retriever comparison (pt.Experiment) with support for:
#  - live PyTerrier transformers (e.g., bi-encoder pipelines, ANN retrievers, etc.)
#  - precomputed runs saved on disk in TREC format
#
# This module intentionally focuses ONLY on building the experiment comparison table.
# --------------------------------------------------------------------------------------


def _read_trec_run(path: Union[str, Path]) -> pd.DataFrame:
    """Read a TREC run file into a DataFrame.

    Expected format (standard TREC):
        qid Q0 docno rank score tag

    Returns a DataFrame with at least: [qid, docno, score, rank].
    """
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Malformed TREC run line in {path}: {line}")
            qid, _q0, docno, rank, score, _tag = parts[:6]
            rows.append((qid, docno, int(rank), float(score)))
    df = pd.DataFrame(rows, columns=["qid", "docno", "rank", "score"])
    df = df.sort_values(["qid", "rank"]).reset_index(drop=True)
    return df


class StaticRunTransformer(pt.Transformer):
    """A PyTerrier Transformer that simply returns a precomputed run.

    Why:
      pt.Experiment expects a list of Transformers. If you already have a run saved
      (e.g., from a dense retriever you executed elsewhere), this wrapper lets you
      compare it alongside other systems using the same evaluation code path.

    Note:
      If the stored run does not include the 'query' column, we merge it from queries.
    """

    def __init__(self, run_df: pd.DataFrame):
        super().__init__()
        required = {"qid", "docno", "score"}
        if not required.issubset(run_df.columns):
            raise ValueError(f"Run DF must contain {sorted(required)}. Found: {list(run_df.columns)}")
        df = run_df.copy()
        if "rank" not in df.columns:
            df = df.sort_values(["qid", "score"], ascending=[True, False])
            df["rank"] = df.groupby("qid").cumcount() + 1
        keep_cols = ["qid", "docno", "score", "rank"]
        if "query" in df.columns:
            keep_cols.append("query")
        self._run_df = df[keep_cols]

    def transform(self, queries: pd.DataFrame) -> pd.DataFrame:
        qids = set(queries["qid"].astype(str).tolist())
        out = self._run_df[self._run_df["qid"].astype(str).isin(qids)].copy()

        if "query" not in out.columns and "query" in queries.columns:
            out = out.merge(queries[["qid", "query"]], on="qid", how="left")

        if "rank" in out.columns:
            out = out.sort_values(["qid", "rank"])
        return out.reset_index(drop=True)


DenseInput = Union[
    pt.Transformer,                 # live retriever pipeline
    Tuple[str, pt.Transformer],      # (name, transformer)
    Tuple[str, Union[str, Path]],    # (name, path to trec run)
]


def build_experiment_comparison_table_for_dense(
    dense_retrievers: Sequence[DenseInput],
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    metrics_dir: Union[str, Path],
    output_csv_name: str = "dense_experiment_table.csv",
) -> pd.DataFrame:
    """Build and save a single pt.Experiment comparison table for dense retrievers."""
    if not pt.java.started():
        pt.java.init()

    metrics_dir_p = ensure_dir(str(metrics_dir))

    names: List[str] = []
    transformers: List[pt.Transformer] = []

    for item in dense_retrievers:
        if isinstance(item, tuple) and len(item) == 2:
            name, obj = item
            if isinstance(obj, (str, Path)):
                run_df = _read_trec_run(obj)
                tr = StaticRunTransformer(run_df)
            else:
                tr = obj  # assume pt.Transformer
            names.append(str(name))
            transformers.append(tr)
        else:
            tr = item  # type: ignore[assignment]
            names.append(getattr(tr, "__name__", tr.__class__.__name__))
            transformers.append(tr)  # type: ignore[arg-type]

    # --- The exact core snippet requested ---
    print("📊 Building experiment comparison table for all baselines")
    exp = pt.Experiment(
        transformers,
        queries,
        qrels,
        eval_metrics=get_measures(),
        names=names,
    )
    if "AP" in exp.columns:
        exp = exp.rename(columns={"AP": "MAP"})
    save_metrics_df(exp, Path(metrics_dir_p) / output_csv_name)
    return exp


def _auto_collect_run_files(runs_dir: Union[str, Path]) -> List[Tuple[str, Path]]:
    """Collect run files from a directory (all *.txt)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")
    run_paths = sorted([p for p in runs_dir.glob("*.txt") if p.is_file()])
    return [(p.stem, p) for p in run_paths]


def main():
    ap = ArgumentParser(description="Build pt.Experiment comparison table for dense retrievers.")
    ap.add_argument("--queries_path", type=str, required=True, help="Path to queries JSON.")
    ap.add_argument("--qrels_path", type=str, required=True, help="Path to qrels JSON.")
    ap.add_argument(
        "--runs_dir",
        type=str,
        default=None,
        help="Directory containing precomputed TREC run files (*.txt). If set, all runs are compared.",
    )
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        help="Add a single run in the format NAME=PATH. Can be repeated.",
    )
    ap.add_argument("--metrics_dir", type=str, required=True, help="Output directory for the CSV table.")
    ap.add_argument(
        "--output_csv_name",
        type=str,
        default="dense_experiment_table.csv",
        help="Output CSV filename.",
    )
    args = ap.parse_args()

    queries_df = load_queries_json(args.queries_path, qid_field="query_id", query_field="question")
    qrels_df = load_qrels_json(args.qrels_path, qid_field="query_id", docno_field="para_id", label_field="relevance")

    dense: List[Tuple[str, Path]] = []
    if args.runs_dir:
        dense.extend(_auto_collect_run_files(args.runs_dir))

    for spec in args.run:
        if "=" not in spec:
            raise ValueError("--run must be in the format NAME=PATH")
        name, path = spec.split("=", 1)
        dense.append((name, Path(path)))

    if not dense:
        raise ValueError(
            "No dense runs provided. Use --runs_dir to load all *.txt runs, or pass one/more --run NAME=PATH."
        )

    build_experiment_comparison_table_for_dense(
        dense_retrievers=dense,
        queries=queries_df,
        qrels=qrels_df,
        metrics_dir=args.metrics_dir,
        output_csv_name=args.output_csv_name,
    )


if __name__ == "__main__":
    main()
