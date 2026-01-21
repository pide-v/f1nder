from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency 'faiss'. Install faiss-cpu or faiss-gpu.") from e

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception as e:
    raise RuntimeError("Missing dependency 'sentence-transformers'. Install sentence-transformers.") from e

from f1nder.utils.io import load_meta, load_queries_json
from f1nder.rerank.run_rerank_biencoder import rerank_run



@dataclass(frozen=True)
class Evidence:
    docno: str
    publication_date: str
    score: float
    text: str


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
#             raise ValueError(f"Expected JSON list in {path}, got {type(obj)}")
#         return obj
#     else:
#         raise ValueError(f"Unsupported extension: {path.suffix}")


# def _load_queries(queries_path: Path, *, qid_field="qid", query_field="query") -> list[tuple[str, str]]:
#     rows = _read_json_or_jsonl(queries_path)
#     out = []
#     for r in rows:
#         out.append((str(r[qid_field]), str(r[query_field])))
#     return out


# def _load_meta(meta_path: Path) -> list[dict]:
#     rows = _read_json_or_jsonl(meta_path)
#     # ensure stable order by internal_id
#     rows = sorted(rows, key=lambda r: int(r["internal_id"]))
#     return rows


def _make_context_text(pub_date: str, text: str, *, prepend_date: bool, date_prefix: str) -> str:
    if prepend_date and pub_date:
        return f"{date_prefix} {pub_date}\n{text}"
    return text


def _retrieve_dense(
    *,
    index,
    meta: list[dict],
    embedder: SentenceTransformer,
    query: str,
    k: int,
    prepend_date: bool,
    date_prefix: str,
) -> list[Evidence]:
    qvec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)

    scores, ids = index.search(qvec, int(k))
    scores = scores[0]
    ids = ids[0]

    evidences: list[Evidence] = []
    for s, iid in zip(scores, ids):
        if iid < 0:
            continue
        m = meta[int(iid)]
        docno = str(m["docno"])
        pub_date = str(m.get("publication_date", "") or "")
        text = str(m.get("text", "") or "")
        evidences.append(
            Evidence(
                docno=docno,
                publication_date=pub_date,
                score=float(s),
                text=_make_context_text(pub_date, text, prepend_date=prepend_date, date_prefix=date_prefix),
            )
        )
    return evidences


# def _rerank(
#     *,
#     reranker: CrossEncoder,
#     query: str,
#     evidences: list[Evidence],
#     batch_size: int = 32,
# ) -> list[Evidence]:
#     pairs = [(query, ev.text) for ev in evidences]
#     scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
#     rescored = [
#         Evidence(docno=ev.docno, publication_date=ev.publication_date, score=float(sc), text=ev.text)
#         for ev, sc in zip(evidences, scores)
#     ]
#     rescored.sort(key=lambda e: e.score, reverse=True)
#     return rescored


def _build_rag_prompt(
    *,
    query: str,
    evidences: list[Evidence],
    max_context_docs: int,
    max_chars_per_doc: int = 1500,
) -> str:
    """
    Why this structure:
    - The model needs a crisp instruction + clearly delimited evidence blocks.
    - We include docno/date to enable traceability (and later citations).
    """
    blocks = []
    for i, ev in enumerate(evidences[:max_context_docs], start=1):
        snippet = ev.text[:max_chars_per_doc]
        blocks.append(
            f"[DOC {i}] docno={ev.docno} date={ev.publication_date} score={ev.score:.4f}\n{snippet}\n"
        )

    context = "\n".join(blocks)
    prompt = (
        "You are a careful assistant. Answer the question using ONLY the evidence provided.\n"
        "If the evidence is insufficient, say you don't know.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"EVIDENCE:\n{context}\n"
        "ANSWER:\n"
    )
    return prompt


def _generate_hf(prompt: str, *, model_name: str, device: str = "cpu", max_new_tokens: int = 256) -> str:
    """
    Local generation via HuggingFace Transformers.
    (Optional dependency: transformers, torch)
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("To use generator='hf', install transformers and torch.") from e

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    mdl.to(device)

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=tok.model_max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)

    # heuristic: return only after "ANSWER:"
    if "ANSWER:" in text:
        return text.split("ANSWER:", 1)[-1].strip()
    return text.strip()


def run_rag(
    *,
    index_path: str | Path,
    meta_path: str | Path,
    queries_path: str | Path,
    answers_out_path: str | Path,
    embedder_model_name: str = "BAAI/bge-base-en-v1.5",
    embedder_device: Optional[str] = None,
    max_length: int = 512,
    k_retrieve: int = 50,
    do_rerank: bool = False,
    reranker_model_name: str = "BAAI/bge-reranker-base",
    reranker_device: Optional[str] = None,
    rerank_topn: int = 20,
    rerank_batch_size: int = 32,
    max_context_docs: int = 5,
    prepend_date: bool = True,
    date_prefix: str = "DATE:",
    generator: str = "none",  # "none" | "hf"
    hf_generator_model: str = "google/flan-t5-base",
    hf_device: str = "cpu",
    max_new_tokens: int = 256,
    save_prompt: bool = False,

) -> dict:
    """
    End-to-end RAG:
      query -> dense retrieve -> (optional rerank) -> prompt -> (optional generate) -> answers.jsonl
    """
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    queries_path = Path(queries_path)
    answers_out_path = Path(answers_out_path)
    answers_out_path.parent.mkdir(parents=True, exist_ok=True)

    index = faiss.read_index(str(index_path))
    meta = load_meta(meta_path)

    embedder = SentenceTransformer(embedder_model_name, device=embedder_device)
    embedder.max_seq_length = int(max_length)

    reranker = None
    if do_rerank:
        reranker = CrossEncoder(reranker_model_name, device=reranker_device)

    queries = load_queries_json(queries_path)

    n_written = 0
    with answers_out_path.open("w", encoding="utf-8") as f:
        for qid, qtext in queries:
            evs = _retrieve_dense(
                index=index,
                meta=meta,
                embedder=embedder,
                query=qtext,
                k=k_retrieve,
                prepend_date=prepend_date,
                date_prefix=date_prefix,
            )

            if do_rerank and reranker is not None:
                evs = rerank_run(
                    run_in_path=None,  # Not used
                    run_out_path=None,  # Not used
                    queries_path=None,  # Not used
                    meta_path=None,  # Not used
                    reranker_model_name=reranker_model_name,
                    device=reranker_device,
                    max_pairs_per_query=rerank_topn,
                    batch_size=rerank_batch_size,
                    prepend_date=prepend_date,
                    date_prefix=date_prefix,
                    )

            prompt = _build_rag_prompt(query=qtext, evidences=evs, max_context_docs=max_context_docs)

            if generator == "none":
                answer = ""
            elif generator == "hf":
                answer = _generate_hf(prompt, model_name=hf_generator_model, device=hf_device, max_new_tokens=max_new_tokens)
            else:
                raise ValueError("generator must be 'none' or 'hf'")

            out = {
                "qid": qid,
                "query": qtext,
                "answer": answer,
                "evidence": [
                    {
                        "docno": ev.docno,
                        "publication_date": ev.publication_date,
                        "score": ev.score,
                        "text_snippet": ev.text[:400],
                    }
                    for ev in evs[:max_context_docs]
                ],
            }
            if save_prompt:
                out["prompt"] = prompt

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_written += 1

    return {
        "answers_out_path": str(answers_out_path),
        "n_queries": len(queries),
        "n_written": n_written,
        "k_retrieve": int(k_retrieve),
        "do_rerank": bool(do_rerank),
        "max_context_docs": int(max_context_docs),
        "generator": generator,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--embedder", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--embedder-device", default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--k", type=int, default=50)

    p.add_argument("--rerank", action="store_true")
    p.add_argument("--reranker", default="BAAI/bge-reranker-base")
    p.add_argument("--reranker-device", default=None)
    p.add_argument("--rerank-topn", type=int, default=20)
    p.add_argument("--rerank-batch", type=int, default=32)

    p.add_argument("--context-docs", type=int, default=5)
    p.add_argument("--no-prepend-date", action="store_true")

    p.add_argument("--generator", default="hf", choices=["none", "hf"])
    p.add_argument("--hf-model", default="google/flan-t5-base")
    p.add_argument("--hf-device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--save-prompt", action="store_true")

    args = p.parse_args()

    info = run_rag(
        index_path=args.index,
        meta_path=args.meta,
        queries_path=args.queries,
        answers_out_path=args.out,
        embedder_model_name=args.embedder,
        embedder_device=args.embedder_device,
        max_length=args.max_length,
        k_retrieve=args.k,
        do_rerank=args.rerank,
        reranker_model_name=args.reranker,
        reranker_device=args.reranker_device,
        rerank_topn=args.rerank_topn,
        rerank_batch_size=args.rerank_batch,
        max_context_docs=args.context_docs,
        prepend_date=not args.no_prepend_date,
        generator=args.generator,
        hf_generator_model=args.hf_model,
        hf_device=args.hf_device,
        max_new_tokens=args.max_new_tokens,
        save_prompt=args.save_prompt,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))