from pathlib import Path
from typing import Tuple
from argparse import ArgumentParser

import pandas as pd
import pyterrier as pt

from f1nder.eval.measures import get_measures
from f1nder.utils.io import save_metrics, read_trec_run


def normalize_qrels_df(qrels: pd.DataFrame) -> pd.DataFrame:
    q = qrels.copy()

    # accetta varianti comuni
    rename = {}
    if "query_id" in q.columns: rename["query_id"] = "qid"
    if "doc_id" in q.columns:   rename["doc_id"] = "docno"
    if "para_id" in q.columns:   rename["para_id"] = "docno"
    if "relevance" in q.columns: rename["relevance"] = "label"
    q = q.rename(columns=rename)

    needed = {"qid", "docno", "label"}
    if not needed.issubset(q.columns):
        raise ValueError(f"Qrels must contain {needed}, got {list(q.columns)}")

    # tipi: IMPORTANTISSIMO per pytrec_eval
    q["qid"] = q["qid"].astype(str)
    q["docno"] = q["docno"].astype(str)
    q["label"] = pd.to_numeric(q["label"], errors="coerce").fillna(0).astype(int)

    return q


def evaluate_run(
    run_path: str | Path,
    qrels: pd.DataFrame,
    metrics=get_measures(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Valuta una run salvata su disco (TREC format) rispetto ai qrels.
    Ritorna: (perquery_df, aggregate_df)
    """
    print(f"🧐 Evaluating run: {run_path} against qrels with {len(qrels)} entries")
    if not pt.java.started():
        pt.java.init()

    results = read_trec_run(run_path)
    qrels = normalize_qrels_df(qrels)

    print(qrels.head())
    print(qrels.columns)
    print(qrels["qid"].head(5).tolist())

    results_perquery = pt.Evaluate(results, qrels, metrics=metrics, perquery=True)
    results_aggregate = pt.Evaluate(results, qrels, metrics=metrics, perquery=False)

    return results_perquery, results_aggregate


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--qrels_file")
    ap.add_argument("--run_file")
    ap.add_argument("--output_file")
    args = ap.parse_args()

    # if isinstance(args.qrels_file, (str, Path)):
    #     if not pt.java.started():
    #         pt.java.init()
    #     qrels = pt.io.read_qrels(str(args.qrels_file))
    # else:
    #     qrels = args.qrels_file

    import json

    def load_qrels_any(path: str | Path) -> pd.DataFrame:
        path = Path(path)

        # 1) JSON (lista di dict) o JSONL
        if path.suffix in {".json", ".jsonl"}:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            if txt.startswith("["):
                data = json.loads(txt)          # JSON list
            else:
                data = [json.loads(line) for line in txt.splitlines() if line.strip()]  # JSONL
            df = pd.DataFrame(data)

            # normalize columns to PyTerrier expectations
            df = df.rename(columns={
                "query_id": "qid",
                "para_id": "docno",
                "relevance": "label",
            })

            return df[["qid", "docno", "label"]]

        # 2) Default: assume TREC qrels file
        if not pt.java.started():
            pt.java.init()
        return pt.io.read_qrels(str(path))
    
    qrels = load_qrels_any(args.qrels_file)

    metrics_perquery, metrics_aggregate = evaluate_run(args.run_file, qrels)

    perquery_output_file = Path(args.output_file).parent / f"perquery_{Path(args.output_file).name}"
    save_metrics(metrics_perquery, perquery_output_file)
    print(f"💾 Saved per-query metrics to {perquery_output_file}")

    aggregate_output_file = Path(args.output_file).parent / f"aggregate_{Path(args.output_file).name}"
    save_metrics(metrics_aggregate, aggregate_output_file)
    print(f"💾 Saved aggregate metrics to {aggregate_output_file}")