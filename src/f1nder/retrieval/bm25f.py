from typing import Dict
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import pyterrier as pt

from f1nder.utils.io import ensure_dir, write_trec_run, save_metrics_df
from f1nder.utils.io import load_queries_json, load_qrels_json
from f1nder.index.index_with_entities import build_or_load_index_from_df
from f1nder.eval.evaluate import evaluate_run, save_metrics
from f1nder.eval.measures import get_measures


def build_bm25f_retriever(indexref: "pt.IndexRef", top_k: int = 100):
    # retrieve BM25F multi-field con pesi
    if not pt.java.started():
        pt.init()

    index = pt.IndexFactory.of(indexref)

    # controlli BM25F: campo 0 = text, campo 1 = entities
    controls = {
        'w.0': 1.0,  # peso text
        'w.1': 1.5,  # peso entities
        'c.0': 0.6,  # normalizzazione text
        'c.1': 0.3   # normalizzazione entities
    }

    return pt.terrier.Retriever(index, wmodel="BM25F", controls=controls, num_results=top_k)


def run_bm25f(
    corpus_path: str,
    queries_path: str,
    qrels_path: str,
    index_dir: str,
    runs_dir: str,
    metrics_dir: str,
):
    if not pt.java.started():
        pt.init()

    # Carica query e qrels
    queries_df = pd.read_json(queries_path)
    queries_df["query"] = queries_df["question"] + " " + queries_df["entities"]
    queries_df.rename(columns={"query_id": "qid", "question": "query"}, inplace=True)

    # Stampa le prime query per controllare il formato
    print(queries_df.head(5))

    qrels_df = load_qrels_json(qrels_path, qid_field="query_id", docno_field="para_id", label_field="relevance")

    # Carica corpus e costruisci indice multi-field
    corpus_df = pd.read_json(corpus_path)
    indexref = build_or_load_index_from_df(corpus_df, index_dir=index_dir)

    runs_dir_p = ensure_dir(runs_dir)
    metrics_dir_p = ensure_dir(metrics_dir)

    run_path = runs_dir_p / "run_bm25f.txt"
    if run_path.exists():
        print(f"⚠️ Run file already exists at {run_path}. Skipping retrieval.")
        results = pt.io.read_results(run_path, format="trec")
        bm25f_retr = pt.Transformer.from_df(results)
    else:
        print("🏃 Running BM25F retrieval with progress...")
        bm25f_retr = build_bm25f_retriever(indexref)
        batch_size = 10

        all_results = []
        for start in tqdm(range(0, len(queries_df), batch_size)):
            batch = queries_df.iloc[start:start+batch_size]
            batch_results = bm25f_retr.transform(batch)
            all_results.append(batch_results)

        results = pd.concat(all_results, ignore_index=True)

        # Salva TREC run
        write_trec_run(results, run_path, run_name="BM25F_entities")
        print(f"✅ Run saved at {run_path}")


    # Valutazione
    metrics_perquery, metrics_aggregate = evaluate_run(run_path, qrels_df)

    save_metrics(metrics_perquery, metrics_dir_p / "metrics_perquery_bm25f.json")
    save_metrics(metrics_aggregate, metrics_dir_p / "metrics_aggregate_bm25f.json")

    print("📊 Building experiment table for BM25F...")

    exp = pt.Experiment(
    [bm25f_retr],
    queries_df,
    qrels_df,
    eval_metrics=get_measures(),
    names=["BM25F"]
    )

    if "AP" in exp.columns:
        exp = exp.rename(columns={"AP": "MAP"})

    save_metrics_df(exp, metrics_dir_p / "bm25f_experiment_table.csv")

    print("✅ BM25F retrieval completed!")
    return metrics_perquery, metrics_aggregate


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--corpus_path", type=str, required=True)
    ap.add_argument("--queries_path", type=str, required=True)
    ap.add_argument("--qrels_path", type=str, required=True)
    ap.add_argument("--index_dir", type=str, required=True)
    ap.add_argument("--runs_dir", type=str, required=True)
    ap.add_argument("--metrics_dir", type=str, required=True)
    args = ap.parse_args()

    run_bm25f(
        corpus_path=args.corpus_path,
        queries_path=args.queries_path,
        qrels_path=args.qrels_path,
        index_dir=args.index_dir,
        runs_dir=args.runs_dir,
        metrics_dir=args.metrics_dir,
    )