from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyterrier as pt
from argparse import ArgumentParser

from f1nder.utils.io import ensure_dir, write_trec_run, save_metrics_df
from f1nder.utils.io import load_corpus_json, load_queries_json, load_qrels_json
from f1nder.index.index import build_or_load_index_from_df
from f1nder.eval.evaluate import evaluate_run, save_metrics
from f1nder.eval.measures import get_measures


def build_baseline_retrievers(indexref: "pt.IndexRef", top_k: int = 1000) -> List[Tuple[str, "pt.Transformer"]]:
    """
    Create 3 simple baselines:
    - TF_IDF
    - BM25
    - PL2 (DFR)
    - DPH (language model with Dirichlet prior smoothing)
    """
    if not pt.java.started():
        pt.java.init()

    index = pt.IndexFactory.of(indexref)

    retrievers: List[Tuple[str, pt.Transformer]] = [
        ("TF_IDF", pt.terrier.Retriever(index, wmodel="TF_IDF", num_results=top_k)),
        ("BM25", pt.terrier.Retriever(index, wmodel="BM25", num_results=top_k)),
        ("PL2", pt.terrier.Retriever(index, wmodel="PL2", num_results=top_k)),
        ("DPH", pt.terrier.Retriever(index, wmodel="DPH", num_results=top_k)),
    ]
    return retrievers


def run_and_evaluate_baselines(
    indexref: "pt.IndexRef",
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    runs_dir: str,
    metrics_dir: str,
    top_k: int = 1000,
) -> Dict[str, pd.DataFrame]:
    """
    Run baselines, save TREC runs, compute per-query and aggregate metrics,
    and also produce a pt.Experiment comparison table.

    Why (pipeline):
    - retrieval -> artifact (run) -> evaluation (perquery + aggregate) -> experiment table
    """

    if not pt.java.started():
        pt.java.init()

    runs_dir_p = ensure_dir(runs_dir)
    metrics_dir_p = ensure_dir(metrics_dir)

    # sanity: required columns
    if not {"qid", "query"}.issubset(set(queries.columns)):
        raise ValueError(f"Queries DF must contain ['qid','query']. Found: {list(queries.columns)}")
    if not {"qid", "docno", "label"}.issubset(set(qrels.columns)):
        raise ValueError(f"Qrels DF must contain ['qid','docno','label']. Found: {list(qrels.columns)}")

    baselines = build_baseline_retrievers(indexref, top_k=top_k)
    outputs: Dict[str, pd.DataFrame] = {}

    # Run each model separately
    for name, retr in baselines:
        print(f"🏃 Running baseline: {name}")
        results = retr.transform(queries)

        # Save run
        run_path = runs_dir_p / f"run_{name}.txt"
        write_trec_run(results, run_path, run_name=name)

        # Evaluate per-query and aggregate
        metrics_perquery, metrics_aggregate = evaluate_run(run_path, qrels)

        # Save metrics
        perquery_metrics_path = metrics_dir_p / f"metrics_perquery_{name}.json"
        aggregate_metrics_path = metrics_dir_p / f"metrics_aggregate_{name}.json"
        save_metrics(metrics_perquery, perquery_metrics_path)
        save_metrics(metrics_aggregate, aggregate_metrics_path)

        outputs[f"{name}_perquery"] = metrics_perquery
        outputs[f"{name}_aggregate"] = metrics_aggregate

    # Single comparison table "like notebook"
    print("📊 Building experiment comparison table for all baselines")
    exp = pt.Experiment(
        [retr for _, retr in baselines],
        queries,
        qrels,
        eval_metrics=get_measures(),
        names=[name for name, _ in baselines],
    )
    if "AP" in exp.columns:
        exp = exp.rename(columns={"AP": "MAP"})
    save_metrics_df(exp, metrics_dir_p / "baselines_experiment_table.csv")
    outputs["experiment_table"] = exp

    return outputs


def run_experiments(
    corpus_path: str,
    queries_path: str,
    qrels_path: str,
    index_dir: str,
    runs_dir: str,
    metrics_dir: str,
):
    if not pt.java.started():
        pt.java.init()

    corpus_df = load_corpus_json(corpus_path, docno_field="para_id", text_field="context")
    queries_df = load_queries_json(queries_path, qid_field="query_id", query_field="question")
    qrels_df = load_qrels_json(qrels_path, qid_field="query_id", docno_field="para_id", label_field="relevance")

    indexref = build_or_load_index_from_df(corpus_df, index_dir=index_dir)

    outputs = run_and_evaluate_baselines(
        indexref=indexref,
        queries=queries_df,
        qrels=qrels_df,
        runs_dir=runs_dir,
        metrics_dir=metrics_dir,
        top_k=1000,
    )
    return outputs


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--corpus_path", type=str)
    ap.add_argument("--queries_path", type=str)
    ap.add_argument("--qrels_path", type=str)
    ap.add_argument("--index_dir", type=str)
    ap.add_argument("--runs_dir", type=str)
    ap.add_argument("--metrics_dir", type=str)
    args = ap.parse_args()

    outputs = run_experiments(
        corpus_path=args.corpus_path,
        queries_path=args.queries_path,
        qrels_path=args.qrels_path,
        index_dir=args.index_dir,
        runs_dir=args.runs_dir,
        metrics_dir=args.metrics_dir,
    )