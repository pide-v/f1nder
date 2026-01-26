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
from f1nder.utils.progress_bar import progress_counter, log
from f1nder.utils.chunking import chunk_text_sliding_window


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


# def _to_dense_text(d: DocRecord, prepend_date: bool, date_prefix: str = "PUBBLICATION_DATE:") -> str:
#     if prepend_date and d.publication_date:
#         return f"{date_prefix} {d.publication_date} TEXT:{d.text}"
#     return d.text


def build_dense_index(
    *,
    corpus_path: str | Path,
    index_out_path: str | Path,
    meta_out_path: str | Path,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 64,
    device: Optional[str] = None,
    max_length: int = 512,
    chunk_tokens: int = 384,
    overlap_tokens: int = 64,
    min_last_chunk_tokens: int = 100,
    prepend_date: bool = True,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict:
    
    print("🚀 Starting build_dense_index...")

    corpus_path = Path(corpus_path)
    index_out_path = Path(index_out_path)
    meta_out_path = Path(meta_out_path)
    index_out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load model
    print(f"🔄 Loading model {model_name} on device {device} with max_length {max_length}...")
    log(f"[build_dense_index] loading model={model_name} device={device}")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = int(max_length)

    # 2) Load corpus -> DocRecord list
    print(f"📚 Loading documents from corpus at {corpus_path}...")
    docs = list(_iter_docs_from_corpus(corpus_path))
    if not docs:
        raise ValueError(f"No documents found in {corpus_path}")

    # 3) Prepare strings to embed (date injected at start)
    print("📝 Preparing texts for embedding...")
    # dense_texts = [_to_dense_text(d, prepend_date=prepend_date, date_prefix=date_prefix) for d in docs]
    chunk_texts: list[str] = []
    chunk_meta: list[dict] = []

    internal_id = 0
    tokenizer = model.tokenizer
    tokenizer.model_max_length = float("inf")  # disable tokenizer-level truncation

    for d in docs:
        # dense_text = _to_dense_text(d, prepend_date=prepend_date, date_prefix=date_prefix)
        dense_text = d.text  # chunking will handle date prepending if needed

        chunks = chunk_text_sliding_window(
            dense_text,
            tokenizer=tokenizer,
            chunk_tokens=int(chunk_tokens),
            overlap_tokens=int(overlap_tokens),
            min_last_chunk_tokens=int(min_last_chunk_tokens),
            date_tokens_budget=16 if prepend_date and d.publication_date else 0,
        )

        for ch in chunks:
            if prepend_date and d.publication_date:
                ch_text = f"PUBBLICATION_DATE: {d.publication_date} TEXT:{ch.text}"
            else:
                ch_text = ch.text

            chunk_texts.append(ch_text)
            chunk_meta.append(
                {
                    "internal_id": internal_id,
                    "docno": d.docno,  # IMPORTANT: resta para_id per la evaluation
                    "chunk_id": int(ch.chunk_id),
                    "start_token": int(ch.start_token),
                    "end_token": int(ch.end_token),
                    "n_tokens": int(ch.n_tokens),
                    "publication_date": d.publication_date,
                    "text": ch.text,   # utile per reranking/RAG
                }
            )
            internal_id += 1
    
    n_docs = len(chunk_texts)
    n_batches = (n_docs + batch_size - 1) // batch_size

    log(f"[build_dense_index] docs={n_docs} batch_size={batch_size} batches={n_batches} max_length={max_length}", verbose=verbose)

    # 4) Encode in batches, but progress is counted in *documents*
    print(f"🔄 Encoding {n_docs} documents in batches of {batch_size}...")
    vectors: list[np.ndarray] = []
    with progress_counter(total=n_docs, desc="Indexing documents", enabled=show_progress) as pbar:
        for batch in _batched(chunk_texts, batch_size):
            emb = model.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,  # we handle progress ourselves
            ).astype(np.float32, copy=False)

            vectors.append(emb)
            # document-level progress:
            pbar.update(len(batch))

    X = np.vstack(vectors)
    n_docs2, dim = X.shape
    if n_docs2 != n_docs:
        raise RuntimeError(f"Embedding count mismatch: got {n_docs2}, expected {n_docs}")

    # 5) Build & save FAISS index
    print(f"💾 Building and saving FAISS index to {index_out_path}...")
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, str(index_out_path))

    # 6) Save metadata mapping (internal_id -> docno + date + text)
    print(f"💾 Saving metadata to {meta_out_path}...")
    # with meta_out_path.open("w", encoding="utf-8") as f:
    #     for i, d in enumerate(docs):
    #         rec = {
    #             "internal_id": i,
    #             "docno": d.docno,
    #             "publication_date": d.publication_date,
    #             "text": d.text,
    #         }
    #         f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with meta_out_path.open("w", encoding="utf-8") as f:
        for rec in chunk_meta:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "n_chunks": int(n_docs),
        "chunk_tokens": int(chunk_tokens),
        "overlap_tokens": int(overlap_tokens),
        "min_last_chunk_tokens": int(min_last_chunk_tokens),
        #"n_docs": n_docs,
        "dim": dim,
        "index_out_path": str(index_out_path),
        "meta_out_path": str(meta_out_path),
        "model_name": model_name,
        "batch_size": batch_size,
        "max_length": max_length,
        "device": device,
        "prepend_date": prepend_date,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--index-out", required=True)
    p.add_argument("--meta-out", required=True)
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--chunk-tokens", type=int, default=384)
    p.add_argument("--overlap-tokens", type=int, default=64)
    p.add_argument("--min-last-chunk-tokens", type=int, default=100)
    p.add_argument("--prepend-date", choices=["true", "false"], default="true")
    p.add_argument("--show-progress", choices=["true", "false"], default="true")
    p.add_argument("--verbose", choices=["true", "false"], default="true")

    args = p.parse_args()

    info = build_dense_index(
        corpus_path=args.corpus,
        index_out_path=args.index_out,
        meta_out_path=args.meta_out,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        min_last_chunk_tokens=args.min_last_chunk_tokens,
        prepend_date=args.prepend_date.lower() == "true",
        show_progress=args.show_progress.lower() == "true",
        verbose=args.verbose.lower() == "true",
        )
    print(json.dumps(info, indent=2, ensure_ascii=False))