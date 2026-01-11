from pathlib import Path
import argparse
import pyterrier as pt
import pandas as pd
from f1nder.utils.io import ensure_dir

def build_or_load_index_from_df(
    corpus_df: pd.DataFrame,
    index_dir: str,
    text_field: str = "text",
    entities_field: str = "entities",
    docno_field: str = "docno",
    force_rebuild: bool = False,
) -> "pt.IndexRef":
    
    if not pt.java.started():
        pt.java.init()

    idx_dir = ensure_dir(index_dir)
    data_props = Path(idx_dir) / "data.properties"

    if data_props.exists() and not force_rebuild:
        return pt.terrier.J.IndexRef.of(str(idx_dir))

    docs_iter = corpus_df[[docno_field, text_field, entities_field]].rename(
        columns={docno_field: "docno", text_field: "text", entities_field: "entities"}
    ).to_dict(orient="records")

    indexer = pt.IterDictIndexer(
        str(idx_dir),
        meta={"docno": 64},
        meta_reverse=["docno"],
        text_attrs=["text", "entities"],
        fields=["text", "entities"]
    )
    indexref = indexer.index(docs_iter)
    return indexref

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25F index with entities")
    parser.add_argument("--document_collection_file", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    corpus_df = pd.read_json(args.document_collection_file)
    
    indexref = build_or_load_index_from_df(
        corpus_df=corpus_df,
        index_dir=args.index_path,
        text_field="text",
        entities_field="entities",
        docno_field="para_id",
        force_rebuild=args.force_rebuild
    )
    print(f"✅ Index built at {args.index_path}")