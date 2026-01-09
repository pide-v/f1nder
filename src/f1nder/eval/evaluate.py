# src/f1nder/eval/evaluate.py

from pathlib import Path
from typing import Tuple
from argparse import ArgumentParser

import pandas as pd
import json
import pyterrier as pt

from f1nder.eval.measures import get_measures
from f1nder.utils.io import save_metrics


def read_trec_run(run_path: str | Path) -> pd.DataFrame:
    """
    Legge una run in formato TREC:
      qid Q0 docno rank score run_name
    e restituisce un DataFrame con colonne compatibili con pt.Evaluate:
      qid, docno, rank, score
    """
    run_path = Path(run_path)
    df = pd.read_csv(
        run_path,
        sep=r"\s+",
        header=None,
        names=["qid", "Q0", "docno", "rank", "score", "run_name"],
        dtype={"qid": str, "Q0": str, "docno": str, "rank": int, "score": float, "run_name": str},
        engine="python",
    )
    return df[["qid", "docno", "rank", "score"]]


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
    if not pt.started():
        pt.init()

    results = read_trec_run(run_path)
    results_perquery = pt.Evaluate(results, qrels, metrics=metrics, perquery=True)
    results_aggregate = pt.Evaluate(results, qrels, metrics=metrics, perquery=False)

    return results_perquery, results_aggregate


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--qrels_file")
    ap.add_argument("--run_file")
    ap.add_argument("--output_file")
    args = ap.parse_args()

    metrics_perquery, metrics_aggregate = evaluate_run(args.run_file, args.qrels_file)

    perquery_output_file = Path(args.output_file).parent / f"perquery_{Path(args.output_file).name}"
    save_metrics(metrics_perquery, perquery_output_file)
    print(f"Saved per-query metrics to {perquery_output_file}")

    aggregate_output_file = Path(args.output_file).parent / f"aggregate_{Path(args.output_file).name}"
    save_metrics(metrics_aggregate, aggregate_output_file)
    print(f"Saved aggregate metrics to {aggregate_output_file}")