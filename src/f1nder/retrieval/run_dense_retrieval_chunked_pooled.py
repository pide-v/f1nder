from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Missing dependency 'faiss'. Install faiss-cpu or faiss-gpu."
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "Missing dependency 'sentence-transformers'. Install sentence-transformers."
    ) from e

from f1nder.utils.io import load_meta, load_queries_json, write_trec_run
from f1nder.utils.progress_bar import progress_counter


def _default_query_prefix_for_model(model_name: str) -> str | None:
    """
    For BGE v1.5 retrieval, docs recommend prefixing queries with an instruction.
    Docs/passages usually do NOT need an instruction prefix.   [oai_citation:1‡Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5?utm_source=chatgpt.com)
    """
    mn = model_name.lower()
    if "bge" in mn and "v1.5" in mn:
        return "Represent this sentence for searching relevant passages: "
    return None


def run_dense_retrieval(
    *,
    index_path: str | Path,
    meta_path: str | Path,
    queries_path: str | Path,
    run_out_path: str | Path,
    model_name: str = "BAAI/bge-base-en-v1.5",
    k: int = 1000,
    chunk_k: int = 5000,
    device: Optional[str] = None,
    max_length: int = 512,
    query_batch_size: int = 64,
    qrels_path: Optional[str | Path] = None,
    show_progress: bool = True,
    progress_desc: str = "Retrieving queries",
    query_prefix: Optional[str] = None,
    pool_docno: bool = True,
    pool_mode: str = "max",  # for now we implement only max
) -> dict:
    """
    Online retrieval: query embeddings -> FAISS search -> TREC run.
    """
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    queries_path = Path(queries_path)
    run_out_path = Path(run_out_path)

    index = faiss.read_index(str(index_path))
    meta = load_meta(meta_path)
    docno_by_internal = [m["docno"] for m in meta]

    print(f"🔁 Loading queries from {queries_path}...")
    # This returns a DataFrame
    queries_df = load_queries_json(
        queries_path=queries_path,
        qid_field="query_id",
        query_field="question",
    )

    # Normalize expected column names (do NOT change the return type)
    # We only rename locally for clarity.
    if "query_id" in queries_df.columns and "question" in queries_df.columns:
        qdf = queries_df.rename(columns={"query_id": "qid", "question": "query"})
    else:
        # If your loader already returns qid/query columns, handle it.
        qdf = queries_df.copy()
        if "qid" not in qdf.columns or "query" not in qdf.columns:
            raise KeyError(
                "Queries DF must have columns ('query_id','question') or ('qid','query'). "
                f"Found: {list(qdf.columns)}"
            )

    qdf["qid"] = qdf["qid"].astype(str)
    qdf["query"] = qdf["query"].astype(str)

    print(f"🔁 Loading model '{model_name}' on device '{device or 'default'}'...")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = int(max_length)

    rows = []
    n_queries = len(qdf)

    print(f"🚀 Encoding and searching {n_queries} queries in batches...")
    # Batch queries for speed (but ranking is still per-query)
    with progress_counter(total=n_queries, desc=progress_desc, enabled=show_progress) as pbar:

        for start in range(0, n_queries, int(query_batch_size)):
            batch_df = qdf.iloc[start : start + int(query_batch_size)]
            qids = batch_df["qid"].tolist()
            qtexts = batch_df["query"].tolist()

            if query_prefix is None:
                query_prefix = _default_query_prefix_for_model(model_name)

            if query_prefix:
                qdf["query"] = query_prefix + qdf["query"]
                print(f"🧠 Using query prefix for retrieval: {query_prefix!r}")
            else:
                print("🧠 No query prefix applied.")

            # Encode all queries in the batch in one go (huge speedup)
            Q = model.encode(
                qtexts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32, copy=False)

            # Search all queries at once: FAISS supports batch search
            scores, internal_ids = index.search(Q, int(k))  # shapes: (B,k), (B,k)

            for i, qid in enumerate(qids):
                if pool_docno:
                    best_score_by_docno: dict[str, float] = {}

                    for s, iid in zip(scores[i], internal_ids[i]):
                        if iid < 0:
                            continue
                        docno = docno_by_internal[int(iid)]
                        s = float(s)

                        # max pooling (robust, standard for passage->doc aggregation)
                        prev = best_score_by_docno.get(docno)
                        if prev is None or s > prev:
                            best_score_by_docno[docno] = s

                    # Sort docnos by pooled score desc, keep top-k (same k as before)
                    ranked = sorted(best_score_by_docno.items(), key=lambda x: x[1], reverse=True)[: int(k)]

                    for rank, (docno, s) in enumerate(ranked, start=1):
                        rows.append({"qid": qid, "docno": docno, "rank": rank, "score": float(s)})

                else:
                    # old behavior (may create duplicate docno per query)
                    rank = 1
                    for s, iid in zip(scores[i], internal_ids[i]):
                        if iid < 0:
                            continue
                        docno = docno_by_internal[int(iid)]
                        rows.append({"qid": qid, "docno": docno, "rank": rank, "score": float(s)})
                        rank += 1


            # Progress “per query” (not per batch)
            pbar.update(len(batch_df))

    print(f"💾 Writing TREC run to {run_out_path}...")
    run_df = pd.DataFrame(rows)
    write_trec_run(results=run_df, run_path=run_out_path, run_name="dense")

    info = {
        "n_queries": int(n_queries),
        "k": int(k),
        "run_out_path": str(run_out_path),
        "model_name": model_name,
        "max_length": int(max_length),
        "query_batch_size": int(query_batch_size),
    }

    if qrels_path is not None:
        info["qrels_path"] = str(qrels_path)
        info["note"] = (
            "qrels_path provided, but evaluation is not executed here. "
            "Use your repo's evaluate_run(...) on the generated run file."
        )

    return info


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--run-out", required=True)
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--k", type=int, default=1000)
    p.add_argument("--chunk-k", type=int, default=5000)
    p.add_argument("--device", default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--qrels", default=None)
    p.add_argument("--show-progress", type=bool, default=True)
    p.add_argument("--progress-desc", type=str, default="Retrieving queries")
    p.add_argument("--query-prefix", default=None)
    p.add_argument("--pool-docno", type=bool, default=True)
    p.add_argument("--pool-mode", type=str, default="max")
    args = p.parse_args()

    info = run_dense_retrieval(
        index_path=args.index,
        meta_path=args.meta,
        queries_path=args.queries,
        run_out_path=args.run_out,
        model_name=args.model,
        k=args.k,
        chunk_k=args.chunk_k,
        device=args.device,
        max_length=args.max_length,
        qrels_path=args.qrels,
        show_progress=args.show_progress,
        progress_desc=args.progress_desc,
        query_prefix=_default_query_prefix_for_model(args.model),
        pool_docno=True,
        pool_mode="max",
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))