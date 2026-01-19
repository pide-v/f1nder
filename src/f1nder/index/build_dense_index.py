from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

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

from f1nder.utils.io import read_json_or_jsonl
from f1nder.utils.progress_bar import progress, log


@dataclass(frozen=True)
class DocRecord:
    """Minimal per-doc metadata we need later for retrieval & run writing."""
    docno: str
    publication_date: str
    text: str  # original context (optionally truncated externally)


def _iter_docs_from_corpus(
    corpus_path: Path,
    *,
    docno_field: str = "para_id",
    text_field: str = "context",
    date_field: str = "publication_date",
) -> Iterator[DocRecord]:
    """
    Converts your corpus records into DocRecord.

    Why this function exists:
    - The dataset schema is "para_id/context/publication_date".
    - We want a predictable internal representation independent of your repo utilities.
    """
    rows = read_json_or_jsonl(corpus_path)
    for r in rows:
        try:
            docno = str(r[docno_field])
            text = str(r[text_field])
            pub_date = str(r.get(date_field, "") or "")
        except KeyError as e:
            raise KeyError(
                f"Corpus record missing expected field {e}. "
                f"Expected fields: {docno_field}, {text_field}, {date_field}"
            ) from e
        yield DocRecord(docno=docno, publication_date=pub_date, text=text)


def _batched(it: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for x in it:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_dense_index(
    *,
    corpus_path: str | Path,
    index_out_path: str | Path,
    meta_out_path: str | Path,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 64,
    device: Optional[str] = None,
    max_length: int = 512,
    prepend_date: bool = True,
    date_prefix: str = "DATE:",
    show_progress: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Offline indexing: corpus -> embeddings -> FAISS index.

    Params:
      - corpus_path: input corpus (.json or .jsonl) with fields para_id/context/publication_date
      - index_out_path: output FAISS index (.faiss)
      - meta_out_path: output JSONL mapping internal_id -> docno (+ optionally more)
        (We store docno and optionally fields to reconstruct context later.)

    Why IndexFlatIP + normalization:
      - If embeddings are L2-normalized, inner product == cosine similarity.
      - FAISS IndexFlatIP supports add/search with those vectors.  [oai_citation:3‡GitHub](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes?utm_source=chatgpt.com)
    """
    corpus_path = Path(corpus_path)
    index_out_path = Path(index_out_path)
    meta_out_path = Path(meta_out_path)

    index_out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_out_path.parent.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(model_name, device=device)
    # max_length controls truncation; important to keep consistent between doc and query encoding
    model.max_seq_length = int(max_length)

    # 1) Load docs
    docs = list(_iter_docs_from_corpus(corpus_path))
    if not docs:
        raise ValueError(f"No documents found in {corpus_path}")

    # 2) Build the exact string we embed (date injected at the beginning)
    def to_dense_text(d: DocRecord) -> str:
        if prepend_date and d.publication_date:
            # putting date first makes it more likely to influence the pooled embedding
            return f"{date_prefix} {d.publication_date}\n{d.text}"
        return d.text

    dense_texts = [to_dense_text(d) for d in docs]

    # 3) Encode in batches -> float32 numpy (faiss expects float32)
    # vectors: list[np.ndarray] = []
    # for batch in _batched(dense_texts, batch_size):
    #     emb = model.encode(
    #         batch,
    #         batch_size=len(batch),
    #         convert_to_numpy=True,
    #         normalize_embeddings=True,  # cosine-ready
    #         show_progress_bar=False,
    #     )
    #     emb = emb.astype(np.float32, copy=False)
    #     vectors.append(emb)

    # X = np.vstack(vectors)
    # if X.ndim != 2:
    #     raise RuntimeError(f"Unexpected embedding shape: {X.shape}")
    # n_docs, dim = X.shape

    # 3) Encode in batches -> float32 numpy (faiss expects float32)
    vectors: list[np.ndarray] = []

    n_docs = len(dense_texts)
    n_batches = (n_docs + batch_size - 1) // batch_size
    log(f"[build_dense_index] docs={n_docs}, batch_size={batch_size}, batches={n_batches}")
    log(f"[build_dense_index] model={model_name}, device={device}, max_length={max_length}")

    for batch in progress(
        _batched(dense_texts, batch_size),
        total=n_batches,
        desc="Embedding docs",
        enabled=show_progress,
        leave=False,
    ):
        emb = model.encode(
            batch,
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine-ready
            show_progress_bar=False,    # IMPORTANT: we handle progress ourselves
        )
        emb = emb.astype(np.float32, copy=False)
        vectors.append(emb)

    X = np.vstack(vectors)
    if X.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape: {X.shape}")
    n_docs, dim = X.shape

    # 4) Build index
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    # 5) Persist index + metadata mapping
    faiss.write_index(index, str(index_out_path))

    # meta.jsonl: line i corresponds to internal id i in FAISS
    # with meta_out_path.open("w", encoding="utf-8") as f:
    #     for i, d in enumerate(docs):
    #         rec = {
    #             "internal_id": i,
    #             "docno": d.docno,
    #             "publication_date": d.publication_date,
    #             # store original text so reranking/RAG later can reconstruct context quickly
    #             "text": d.text,
    #         }
    #         f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # meta.jsonl: line i corresponds to internal id i in FAISS
    with meta_out_path.open("w", encoding="utf-8") as f:
        for i, d in progress(
            enumerate(docs),
            total=len(docs),
            desc="Writing meta",
            enabled=show_progress,
            leave=False,
        ):
            rec = {
                "internal_id": i,
                "docno": d.docno,
                "publication_date": d.publication_date,
                "text": d.text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "n_docs": n_docs,
        "dim": dim,
        "index_out_path": str(index_out_path),
        "meta_out_path": str(meta_out_path),
        "model_name": model_name,
        "max_length": max_length,
    }


if __name__ == "__main__":
    # This script is designed to be imported and called via build_dense_index(...)
    # Keep CLI minimal by design.
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--index-out", required=True)
    p.add_argument("--meta-out", required=True)
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--prepend-date", type=bool, default=True)
    p.add_argument("--show-progress", type=bool, default=True)
    p.add_argument("--verbose", type=bool, default=True
                   )

    args = p.parse_args()

    info = build_dense_index(
        corpus_path=args.corpus,
        index_out_path=args.index_out,
        meta_out_path=args.meta_out,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
        prepend_date=args.prepend_date,
        show_progress=args.show_progress,
        verbose=args.verbose,
        )
    print(json.dumps(info, indent=2, ensure_ascii=False))