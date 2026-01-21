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


# def _read_json_or_jsonl(path: Path) -> list[dict]:
#     suffix = path.suffix.lower()
#     if suffix == ".jsonl":
#         rows = []
#         with path.open("r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 rows.append(json.loads(line))
#         return rows
#     elif suffix == ".json":
#         with path.open("r", encoding="utf-8") as f:
#             obj = json.load(f)
#         if not isinstance(obj, list):
#             raise ValueError(f"Expected a JSON list in {path}, got: {type(obj)}")
#         return obj
#     else:
#         raise ValueError(f"Unsupported file extension: {path.suffix}. Use .json or .jsonl")


# def _load_meta(meta_path: Path) -> list[dict]:
#     # meta.jsonl written by build_dense_index
#     rows = _read_json_or_jsonl(meta_path)
#     # if someone saved meta as json list, it works too.
#     # we expect internal_id contiguous from 0..N-1
#     rows_sorted = sorted(rows, key=lambda r: int(r["internal_id"]))
#     return rows_sorted


# def _load_queries_json(
#     queries_path: Path,
#     *,
#     qid_field: str = "qid",
#     text_field: str = "query",
# ) -> list[tuple[str, str]]:
#     """
#     Expects queries JSON/JSONL with at least (qid, query).
#     If your repo uses different names, just override qid_field/text_field.
#     """
#     rows = _read_json_or_jsonl(queries_path)
#     out: list[tuple[str, str]] = []
#     for r in rows:
#         try:
#             qid = str(r[qid_field])
#             qtext = str(r[text_field])
#         except KeyError as e:
#             raise KeyError(
#                 f"Queries record missing expected field {e}. "
#                 f"Expected fields: {qid_field}, {text_field}"
#             ) from e
#         out.append((qid, qtext))
#     if not out:
#         raise ValueError(f"No queries found in {queries_path}")
#     return out


# def _write_trec_run(
#     run_df: pd.DataFrame,
#     run_path: Path,
#     *,
#     system_name: str = "dense",
# ) -> None:
#     """
#     Writes a TREC run file:
#       qid Q0 docno rank score system
#     """
#     run_path.parent.mkdir(parents=True, exist_ok=True)
#     needed = {"qid", "docno", "rank", "score"}
#     missing = needed - set(run_df.columns)
#     if missing:
#         raise ValueError(f"run_df missing columns: {missing}")

#     with run_path.open("w", encoding="utf-8") as f:
#         for row in run_df.itertuples(index=False):
#             f.write(f"{row.qid} Q0 {row.docno} {row.rank} {row.score} {system_name}\n")


# def run_dense_retrieval(
#     *,
#     index_path: str | Path,
#     meta_path: str | Path,
#     queries_path: str | Path,
#     run_out_path: str | Path,
#     model_name: str = "BAAI/bge-base-en-v1.5",
#     k: int = 100,
#     device: Optional[str] = None,
#     max_length: int = 512,
#     # Optional evaluation hooks (if you want to plug your repo eval later)
#     qrels_path: Optional[str | Path] = None,
# ) -> dict:
#     """
#     Online retrieval: query embeddings -> FAISS search -> TREC run.

#     All required paths are explicit args, as requested:
#       - index_path: FAISS index produced offline
#       - meta_path: meta.jsonl produced offline (internal_id -> docno)
#       - queries_path: queries json/jsonl
#       - run_out_path: where to write the trec run

#     Why we keep meta_path:
#       - FAISS returns internal ids; we need to map them back to docno used by qrels.  [oai_citation:4‡GitHub](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes?utm_source=chatgpt.com)
#     """
#     index_path = Path(index_path)
#     meta_path = Path(meta_path)
#     queries_path = Path(queries_path)
#     run_out_path = Path(run_out_path)

#     index = faiss.read_index(str(index_path))
#     meta = load_meta(meta_path)
#     docno_by_internal = [m["docno"] for m in meta]

#     queries = load_queries_json(queries_path=queries_path, qid_field="query_id", query_field="question")

#     model = SentenceTransformer(model_name, device=device)
#     model.max_seq_length = int(max_length)

#     rows = []
#     for qid, qtext in queries:
#         qvec = model.encode(
#             [qtext],
#             convert_to_numpy=True,
#             normalize_embeddings=True,
#             show_progress_bar=False,
#         ).astype(np.float32, copy=False)

#         # FAISS returns (scores, internal_ids)
#         scores, internal_ids = index.search(qvec, int(k))
#         scores = scores[0]
#         internal_ids = internal_ids[0]

#         # Build run rows
#         rank = 1
#         for s, iid in zip(scores, internal_ids):
#             if iid < 0:
#                 continue  # faiss uses -1 for empty
#             docno = docno_by_internal[int(iid)]
#             rows.append(
#                 {
#                     "qid": qid,
#                     "docno": docno,
#                     "rank": rank,
#                     "score": float(s),
#                 }
#             )
#             rank += 1

#     run_df = pd.DataFrame(rows)
#     write_trec_run(run_df, run_out_path, system_name="dense")

#     # Optional: you can plug your own evaluation here.
#     # I keep the hook to respect your pipeline, but I don't assume your repo functions.
#     info = {
#         "n_queries": len(queries),
#         "k": int(k),
#         "run_out_path": str(run_out_path),
#         "model_name": model_name,
#         "max_length": max_length,
#     }

#     if qrels_path is not None:
#         info["qrels_path"] = str(qrels_path)
#         info["note"] = (
#             "qrels_path provided, but evaluation is not executed in this standalone script. "
#             "Use your repo's evaluate_run(...) on the generated run file."
#         )

#     return info


def run_dense_retrieval(
    *,
    index_path: str | Path,
    meta_path: str | Path,
    queries_path: str | Path,
    run_out_path: str | Path,
    model_name: str = "BAAI/bge-base-en-v1.5",
    k: int = 100,
    device: Optional[str] = None,
    max_length: int = 512,
    query_batch_size: int = 64,
    qrels_path: Optional[str | Path] = None,
    show_progress: bool = True,
    progress_desc: str = "Retrieving queries",
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
                rank = 1
                for s, iid in zip(scores[i], internal_ids[i]):
                    if iid < 0:
                        continue
                    docno = docno_by_internal[int(iid)]
                    rows.append(
                        {"qid": qid, "docno": docno, "rank": rank, "score": float(s)}
                    )
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
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--qrels", default=None)
    p.add_argument("--show-progress", type=bool, default=True)
    p.add_argument("--progress-desc", type=str, default="Retrieving queries")
    args = p.parse_args()

    info = run_dense_retrieval(
        index_path=args.index,
        meta_path=args.meta,
        queries_path=args.queries,
        run_out_path=args.run_out,
        model_name=args.model,
        k=args.k,
        device=args.device,
        max_length=args.max_length,
        qrels_path=args.qrels,
        show_progress=args.show_progress,
        progress_desc=args.progress_desc,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))