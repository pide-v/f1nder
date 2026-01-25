import json
from pathlib import Path
from typing import Any, Dict, List

import spacy

from f1nder.utils.io import read_json_or_jsonl


nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


# --- preprocess the corpus --- #
def preprocess_text(text: str) -> str:
    """
    Light IR-style preprocessing:
    - tokenization
    - lowercasing
    - punctuation removal
    """
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue

        tokens.append(token.lower_.strip())

    return " ".join(tokens)


def preprocess_collection(
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Legge una document collection JSON/JSONL,
    applica il pre-processing al campo 'context'
    e salva un nuovo campo 'pre_processed'.
    """
    documents: List[Dict[str, Any]] = read_json_or_jsonl(input_path)

    for doc in documents:
        context = doc.get("context", "")
        doc["preprocessed"] = preprocess_text(context)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)


# --- preprocess the queries --- #
def preprocess_query(text: str) -> str:
    """
    Light IR-style query preprocessing (same as corpus):
    - tokenization
    - lowercasing
    - punctuation removal
    """
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue

        tokens.append(token.lower_.strip())

    return " ".join(tokens)


def preprocess_queries(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add a 'pre_processed' field to each query dict,
    using the same preprocessing pipeline as the corpus.
    """
    for q in queries:
        question = q.get("question", "")
        q["preprocessed"] = preprocess_query(question)

    return queries


def save_queries_json(
    queries: List[Dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Save queries (with pre_processed field) to a JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-process a document collection.")
    parser.add_argument("--input_corpus_file", type=str, required=True, help="Path to the input JSON/JSONL file.")
    parser.add_argument("--output_corpus_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--input_queries_file", type=str, help="Path to the input queries JSON/JSONL file.")
    parser.add_argument("--output_queries_file", type=str, help="Path to the output queries JSON file.")
    args = parser.parse_args()

    preprocess_collection(args.input_corpus_file, args.output_corpus_file)

    queries = read_json_or_jsonl(args.input_queries_file)
    queries = preprocess_queries(queries)
    save_queries_json(queries, args.output_queries_file)