from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    from sentence_transformers import CrossEncoder
except Exception as e:
    raise RuntimeError(
        "Missing dependency 'sentence-transformers'. Install sentence-transformers."
    ) from e

from f1nder.utils.io import (
    load_queries_json,
    load_meta_docno_to_text,
    read_trec_run,
    write_trec_run,
)


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
#         raise ValueError(f"Unsupported extension: {path.suffix} (use .json/.jsonl)")


# def _load_queries_json(
#     queries_path: Path,
#     *,
#     qid_field: str = "qid",
#     text_field: str = "query",
# ) -> dict[str, str]:
#     rows = _read_json_or_jsonl(queries_path)
#     out: dict[str, str] = {}
#     for r in rows:
#         if qid_field not in r or text_field not in r:
#             raise KeyError(
#                 f"Queries missing fields. Expected '{qid_field}' and '{text_field}'."
#             )
#         out[str(r[qid_field])] = str(r[text_field])
#     if not out:
#         raise ValueError(f"No queries loaded from {queries_path}")
#     return out


# def _load_meta_docno_to_text(
#     meta_path: Path,
#     *,
#     prepend_date: bool = True,
#     date_prefix: str = "DATE:",
# ) -> dict[str, str]:
#     """
#     meta.jsonl comes from build_dense_index.py.
#     We construct the reranker 'context string' consistently:
#       DATE: <publication_date>\n<text>
#     """
#     rows = _read_json_or_jsonl(meta_path)
#     out: dict[str, str] = {}
#     for r in rows:
#         docno = str(r["docno"])
#         text = str(r.get("text", "") or "")
#         pub_date = str(r.get("publication_date", "") or "")
#         if prepend_date and pub_date:
#             ctx = f"{date_prefix} {pub_date}\n{text}"
#         else:
#             ctx = text
#         out[docno] = ctx
#     if not out:
#         raise ValueError(f"No meta loaded from {meta_path}")
#     return out


# def _read_trec_run(run_path: Path) -> pd.DataFrame:
#     """
#     Expects lines:
#       qid Q0 docno rank score system
#     """
#     rows = []
#     with run_path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.split()
#             if len(parts) < 6:
#                 raise ValueError(f"Bad TREC run line: {line}")
#             qid, _q0, docno, rank, score, system = parts[:6]
#             rows.append(
#                 {
#                     "qid": qid,
#                     "docno": docno,
#                     "rank": int(rank),
#                     "score": float(score),
#                     "system": system,
#                 }
#             )
#     df = pd.DataFrame(rows)
#     if df.empty:
#         raise ValueError(f"Empty run file: {run_path}")
#     return df


# def _write_trec_run(
#     run_df: pd.DataFrame,
#     run_path: Path,
#     *,
#     system_name: str,
# ) -> None:
#     run_path.parent.mkdir(parents=True, exist_ok=True)
#     with run_path.open("w", encoding="utf-8") as f:
#         for row in run_df.itertuples(index=False):
#             f.write(f"{row.qid} Q0 {row.docno} {row.rank} {row.score} {system_name}\n")


def rerank_run(
    *,
    run_in_path: str | Path,
    run_out_path: str | Path,
    queries_path: str | Path,
    meta_path: str | Path,
    reranker_model_name: str = "BAAI/bge-reranker-base",
    device: Optional[str] = None,
    max_pairs_per_query: Optional[int] = None,
    batch_size: int = 32,
    prepend_date: bool = True,
    date_prefix: str = "DATE:",
    system_name: str = "dense_rerank",
) -> dict:
    """
    Rerank top-k candidates per query with a cross-encoder.

    Paths (explicit, as requested):
      - run_in_path: input TREC run (dense retrieval output)
      - run_out_path: output TREC run (reranked)
      - queries_path: queries JSON/JSONL (qid -> query text)
      - meta_path: meta JSONL from indexing (docno -> text (+ date))
    """
    run_in_path = Path(run_in_path)
    run_out_path = Path(run_out_path)
    queries_path = Path(queries_path)
    meta_path = Path(meta_path)

    qmap = load_queries_json(queries_path)
    docmap = load_meta_docno_to_text(meta_path, prepend_date=prepend_date, date_prefix=date_prefix)
    run_df = read_trec_run(run_in_path)

    # Optional: cap how many candidates per qid we rerank (top-N by existing rank)
    if max_pairs_per_query is not None:
        run_df = (
            run_df.sort_values(["qid", "rank"], ascending=[True, True])
            .groupby("qid", as_index=False)
            .head(int(max_pairs_per_query))
        )

    # Build all (query, doc) pairs in the same order as run_df rows
    pairs = []
    missing_q = 0
    missing_d = 0

    for r in run_df.itertuples(index=False):
        qtext = qmap.get(r.qid)
        if qtext is None:
            missing_q += 1
            qtext = ""  # keep alignment; will score poorly
        dtext = docmap.get(r.docno)
        if dtext is None:
            missing_d += 1
            dtext = ""
        pairs.append((qtext, dtext))

    # Cross-encoder scoring
    model = CrossEncoder(reranker_model_name, device=device)
    scores = model.predict(pairs, batch_size=int(batch_size), show_progress_bar=True)

    # Attach new scores and rerank within each query
    run_df = run_df.copy()
    run_df["rerank_score"] = scores

    run_df = run_df.sort_values(["qid", "rerank_score"], ascending=[True, False])

    # Recompute ranks 1..n per qid
    run_df["rank"] = run_df.groupby("qid").cumcount() + 1
    run_df["score"] = run_df["rerank_score"].astype(float)
    run_df = run_df[["qid", "docno", "rank", "score"]]

    write_trec_run(run_df, run_out_path, system_name=system_name)

    return {
        "run_in_path": str(run_in_path),
        "run_out_path": str(run_out_path),
        "reranker_model_name": reranker_model_name,
        "pairs_scored": int(len(pairs)),
        "max_pairs_per_query": max_pairs_per_query,
        "missing_queries": missing_q,
        "missing_docs": missing_d,
        "system_name": system_name,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run-in", required=True)
    p.add_argument("--run-out", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--model", default="BAAI/bge-reranker-base")
    p.add_argument("--device", default=None)
    p.add_argument("--topn", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-prepend-date", action="store_true")
    args = p.parse_args()

    info = rerank_run(
        run_in_path=args.run_in,
        run_out_path=args.run_out,
        queries_path=args.queries,
        meta_path=args.meta,
        reranker_model_name=args.model,
        device=args.device,
        max_pairs_per_query=args.topn,
        batch_size=args.batch_size,
        prepend_date=not args.no_prepend_date,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))