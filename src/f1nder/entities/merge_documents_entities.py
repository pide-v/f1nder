import json
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict, Any

# prende document_collection.json che ha come campi: para_id, context, raw_ocr ...
# prende document_collection_entities.json che ha i campi: para_id e entities.
# crea file document_collection_merged.json che ha come campi: para_id, text e entities.
# sarà poi indicizzato e utilizzato per retrieval con BM25F

def merge_documents_entities(
    docs_file: str,
    entities_file: str,
    output_file: str,
) -> None:

    docs_path = Path(docs_file)
    ents_path = Path(entities_file)
    out_path = Path(output_file)

    with docs_path.open("r", encoding="utf-8") as f:
        documents: List[Dict[str, Any]] = json.load(f)

    with ents_path.open("r", encoding="utf-8") as f:
        entities: List[Dict[str, Any]] = json.load(f)

    # lookup for entities: para_id -> list of entities
    entities_lookup = {item["para_id"]: item["entities"] for item in entities}

    merged = []
    for doc in documents:
        para_id = doc.get("para_id")
        text = doc.get("context", "")  # prendo solo 'context' as text

        if not para_id:
            continue

        doc_entities = entities_lookup.get(para_id, [])

        merged.append({
            "para_id": para_id,
            "text": text,
            "entities": doc_entities
        })

    # check output folder exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # salva
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(merged)} merged documents to {out_path}")
    print(json.dumps(merged[:3], indent=2))


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--input_doc_file", required=True)
    ap.add_argument("--input_entities_file", required=True)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    merge_documents_entities(args.input_doc_file, args.input_entities_file, args.output_file)
