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
    # if "AP" in results_aggregate:
    #     results_perquery["MAP"] = results_perquery["AP"]
    #     del results_perquery["AP"]

    return results_perquery, results_aggregate


# def evaluate_run(
#     run_file: Path,
#     qrels_file: Path,
# ) -> Dict[str, float]:

#     # if not run_file.exists():
#     #     raise FileNotFoundError(f"Run file not found: {run_file}")

#     # if not qrels_file.exists():
#     #     raise FileNotFoundError(f"Qrels file not found: {qrels_file}")

#     # Make sure PyTerrier is initialized (safe to call multiple times)
#     if not pt.java.started():
#         pt.java.init()

#     measures = get_measures()

#     # pt.evaluate returns a dict-like object
#     results_perquery = pt.Evaluate(
#         run=run_file,
#         qrels=qrels_file,
#         metrics=measures,
#         perquery=True,
#     )

#     results_aggregate = pt.Evaluate(
#         run=run_file,
#         qrels=qrels_file,
#         metrics=measures,
#         perquery=False,
#     )

#     # Convert metric objects to readable strings
#     metrics_dict_perquery = {}
#     for m, v in results_perquery.items():
#         metrics_dict_perquery[str(m)] = float(v)
    
#     metrics_dict_aggregate = {}
#     for m, v in results_aggregate.items():
#         metrics_dict_aggregate[str(m)] = float(v)

#     return metrics_dict_perquery, metrics_dict_aggregate


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