from pathlib import Path

import pyterrier as pt
import pandas as pd

from f1nder.utils.io import ensure_dir


def build_or_load_index_from_df(
    corpus_df: pd.DataFrame,
    index_dir: str,
    text_field: str = "text",
    docno_field: str = "docno",
    force_rebuild: bool = False,
) -> "pt.IndexRef":

    if not pt.java.started():
        pt.java.init()

    idx_dir = ensure_dir(index_dir)
    data_props = Path(idx_dir) / "data.properties"

    # if already return an IndexRef, not an Index
    if data_props.exists() and not force_rebuild:
        return pt.terrier.J.IndexRef.of(str(idx_dir))

    # else build a new Index
    docs_iter = corpus_df[[docno_field, text_field]].rename(
        columns={docno_field: "docno", text_field: "text"}
    ).to_dict(orient="records")

    indexer = pt.IterDictIndexer(
        str(idx_dir),
        meta={"docno": 64},
        meta_reverse=["docno"],
        text_attrs=["text"],
    )
    indexref = indexer.index(docs_iter)
    return indexref


if __name__ == "__main__":
    import argparse
    from f1nder.utils.io import load_corpus_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--document_collection_file", type=str, required=True, help="Path to the document collection JSONL file.")
    parser.add_argument("--index_path", type=str, required=True, help="Path to save the built index.")
    parser.add_argument("--text_field", type=str, default="text", help="Field name for document text.")
    parser.add_argument("--docno_field", type=str, default="docno", help="Field name for document identifier.")
    parser.add_argument("--force_rebuild", type=bool, default=False, help="Force rebuilding the index even if it exists.")
    args = parser.parse_args()

    corpus_df = load_corpus_json(args.document_collection_file, docno_field=args.docno_field, text_field=args.text_field)

    build_or_load_index_from_df(
        corpus_df=corpus_df,
        index_dir=args.index_path,
        text_field="text",
        docno_field="docno",
        force_rebuild=args.force_rebuild,
    )