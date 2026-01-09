import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json_or_jsonl(path: str | Path) -> Any:
    """
    Read JSON (array/object) or JSONL (one JSON object per line).

    Why:
    - IR datasets often ship as JSONL for streaming + big files.
    - Some are normal JSON arrays. This supports both.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    # Try standard JSON first
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Fallback to JSONL
        items: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {i} in {p}: {e}") from e
        return items


def _pick_first_existing_key(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    """Return first key in `keys` that exists in dict `d`."""
    for k in keys:
        if k in d:
            return k
    return None


def load_corpus_json(
    corpus_path: str | Path,
    docno_field: Optional[str] = None,
    text_field: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a corpus from JSON/JSONL and return DataFrame with columns: ['docno', 'text'].

    Expected inputs:
    - JSON array of objects: [{...}, {...}]
    - JSONL: one object per line

    Field inference (if not provided):
    - docno_field candidates: docno, doc_id, docid, id, _id
    - text_field candidates: text, body, contents, content, document, passage

    Why:
    - PyTerrier's IterDictIndexer wants 'docno' plus text attrs.
    """
    data = read_json_or_jsonl(corpus_path)
    if isinstance(data, dict):
        # sometimes corpus is wrapped, e.g. {"documents":[...]}
        # pick the first list-like field if possible
        list_candidates = [v for v in data.values() if isinstance(v, list)]
        if len(list_candidates) == 1:
            data = list_candidates[0]
        else:
            raise ValueError(
                f"Corpus JSON root is an object, not a list. "
                f"Provide a JSON list or JSONL. File: {corpus_path}"
            )

    if not isinstance(data, list) or not data:
        raise ValueError(f"Corpus must be a non-empty list of objects. File: {corpus_path}")

    first = data[0]
    if not isinstance(first, dict):
        raise ValueError(f"Corpus items must be JSON objects/dicts. File: {corpus_path}")

    if docno_field is None:
        docno_field = _pick_first_existing_key(first, ["docno", "doc_id", "docid", "id", "_id"])
    if text_field is None:
        text_field = _pick_first_existing_key(first, ["text", "body", "contents", "content", "document", "passage"])

    if docno_field is None or text_field is None:
        raise ValueError(
            f"Could not infer docno/text fields from corpus. "
            f"First item keys: {list(first.keys())}. "
            f"Pass docno_field and text_field explicitly."
        )

    df = pd.DataFrame(data)
    if docno_field not in df.columns or text_field not in df.columns:
        raise ValueError(
            f"Corpus missing required fields. Needed: {docno_field}, {text_field}. "
            f"Found columns: {list(df.columns)}"
        )

    out = df[[docno_field, text_field]].copy()
    out.columns = ["docno", "text"]
    out["docno"] = out["docno"].astype(str)
    out["text"] = out["text"].astype(str)
    return out


def load_queries_json(
    queries_path: str | Path,
    qid_field: Optional[str] = None,
    query_field: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load queries/topics from JSON/JSONL -> DataFrame with ['qid','query'].

    Field inference:
    - qid: qid, query_id, id, topic
    - query: query, text, title
    """
    data = read_json_or_jsonl(queries_path)
    if isinstance(data, dict):
        list_candidates = [v for v in data.values() if isinstance(v, list)]
        if len(list_candidates) == 1:
            data = list_candidates[0]
        else:
            raise ValueError(f"Queries JSON root is an object, not a list. File: {queries_path}")

    if not isinstance(data, list) or not data:
        raise ValueError(f"Queries must be a non-empty list of objects. File: {queries_path}")

    first = data[0]
    if not isinstance(first, dict):
        raise ValueError(f"Queries items must be JSON objects/dicts. File: {queries_path}")

    if qid_field is None:
        qid_field = _pick_first_existing_key(first, ["qid", "query_id", "id", "topic"])
    if query_field is None:
        query_field = _pick_first_existing_key(first, ["query", "text", "title"])

    if qid_field is None or query_field is None:
        raise ValueError(
            f"Could not infer qid/query fields. First item keys: {list(first.keys())}. "
            f"Pass qid_field and query_field explicitly."
        )

    df = pd.DataFrame(data)
    out = df[[qid_field, query_field]].copy()
    out.columns = ["qid", "query"]
    out["qid"] = out["qid"].astype(str)
    out["query"] = out["query"].astype(str)
    return out


def load_qrels_json(
    qrels_path: str | Path,
    qid_field: Optional[str] = None,
    docno_field: Optional[str] = None,
    label_field: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load qrels from JSON/JSONL -> DataFrame with ['qid','docno','label'].

    Field inference:
    - qid: qid, query_id, topic
    - docno: docno, doc_id, docid, id, para_id
    - label: label, rel, relevance, judgement, score
    """
    data = read_json_or_jsonl(qrels_path)
    if isinstance(data, dict):
        list_candidates = [v for v in data.values() if isinstance(v, list)]
        if len(list_candidates) == 1:
            data = list_candidates[0]
        else:
            raise ValueError(f"Qrels JSON root is an object, not a list. File: {qrels_path}")

    if not isinstance(data, list) or not data:
        raise ValueError(f"Qrels must be a non-empty list of objects. File: {qrels_path}")

    first = data[0]
    if not isinstance(first, dict):
        raise ValueError(f"Qrels items must be JSON objects/dicts. File: {qrels_path}")

    if qid_field is None:
        qid_field = _pick_first_existing_key(first, ["qid", "query_id", "topic"])
    if docno_field is None:
        docno_field = _pick_first_existing_key(first, ["docno", "doc_id", "docid", "id", "para_id"])
    if label_field is None:
        label_field = _pick_first_existing_key(first, ["label", "rel", "relevance", "judgement", "score"])

    if qid_field is None or docno_field is None or label_field is None:
        raise ValueError(
            f"Could not infer qrels fields. First item keys: {list(first.keys())}. "
            f"Pass qid_field/docno_field/label_field explicitly."
        )

    df = pd.DataFrame(data)
    out = df[[qid_field, docno_field, label_field]].copy()
    out.columns = ["qid", "docno", "label"]
    out["qid"] = out["qid"].astype(str)
    out["docno"] = out["docno"].astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    if out["label"].isna().any():
        raise ValueError("Some qrels labels could not be parsed as numeric.")
    return out


def write_trec_run(results: pd.DataFrame, run_path: str | Path, run_name: str) -> None:
    """
    Save results in standard TREC run format:
      qid Q0 docno rank score run_name
    """
    p = Path(run_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    needed = {"qid", "docno", "rank", "score"}
    missing = needed - set(results.columns)
    if missing:
        raise ValueError(f"Results missing columns {missing}. Found: {list(results.columns)}")

    out = results[["qid", "docno", "rank", "score"]].copy()
    out["Q0"] = "Q0"
    out["run_name"] = run_name
    out = out[["qid", "Q0", "docno", "rank", "score", "run_name"]]

    out["rank"] = out["rank"].astype(int)
    out["score"] = out["score"].astype(float)

    out.to_csv(p, sep=" ", header=False, index=False, encoding="utf-8")


def save_metrics_df(df: pd.DataFrame, out_path: str | Path) -> None:
    """Save metrics DataFrame as CSV (stable and human-readable)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8")


def save_metrics(
    metrics: Dict[str, float],
    output_path: Path
) -> None:
    """
    Save metrics dictionary to JSON.

    Parameters
    ----------
    metrics : Dict[str, float]
        Evaluation metrics
    output_path : Path
        Where to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)