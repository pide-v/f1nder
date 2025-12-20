import json
import os
import re
import unicodedata
import string
"""
inputs = ["train.json", "validation.json", "test.json"]
output = "document_collection.json"

def load_list_or_empty(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"Skipping {path} because it is missing or empty")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        print(f"Skipping {path} because it is not a list at the top level")
        return []
    except json.JSONDecodeError:
        print(f"Skipping {path} because it is not valid JSON")
        return []

def project(recs):
    out = []
    for r in recs:
        out.append({
            "para_id": r.get("para_id", ""),
            "context": r.get("context", ""),
            "raw_ocr": r.get("raw_ocr", ""),
            "publication_date": r.get("publication_date", "")
        })
    return out

all_recs = []
for p in inputs:
    recs = load_list_or_empty(p)
    print(f"Loaded {len(recs)} records from {p}")
    all_recs.extend(project(recs))

# deduplicate by para_id keeping the first one seen
uniq = {}
for rec in all_recs:
    pid = rec.get("para_id", "")
    if pid and pid not in uniq:
        uniq[pid] = rec

result = list(uniq.values())

with open(output, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(result)} records to {output}")
print(json.dumps(result[:3], indent=2))
"""

input_file = "test.json"
output_file = "test_queries.json"

# Load the data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

def clean_question(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    return text.strip()

# Extract and clean
queries = [
    {
        "query_id": item.get("query_id", ""),
        "question": clean_question(item.get("question", "")),
    }
    for item in data
]

# Sort by query_id (assuming numeric)
queries = sorted(queries, key=lambda x: int(x["query_id"]) if str(x["query_id"]).isdigit() else x["query_id"])

# Keep only the first 10,000
queries = queries[:10000]

# Save new JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(queries, f, ensure_ascii=False, indent=2)

print(f"Saved {len(queries)} entries to {output_file}")
print(json.dumps(queries[:3], indent=2))

input_file = "test.json"
qrels_file = "test_qrels.json"
answers_file = "test_query_answers.json"

# Load the data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Build the qrels file: query_id, iteration=0, para_id, relevance=1
qrels = [
    {
        "query_id": item.get("query_id", ""),
        "iteration": 0,
        "para_id": item.get("para_id", ""),
        "relevance": 1
    }
    for item in data
]

# Build the query_answers file: same plus answer and org_answer
query_answers = [
    {
        "query_id": item.get("query_id", ""),
        "iteration": 0,
        "para_id": item.get("para_id", ""),
        "relevance": 1,
        "answer": item.get("answer", ""),
        "org_answer": item.get("org_answer", "")
    }
    for item in data
]

# Save both files
with open(qrels_file, "w", encoding="utf-8") as f:
    json.dump(qrels, f, ensure_ascii=False, indent=2)

with open(answers_file, "w", encoding="utf-8") as f:
    json.dump(query_answers, f, ensure_ascii=False, indent=2)

print(f"Saved {len(qrels)} entries to {qrels_file}")
print(f"Saved {len(query_answers)} entries to {answers_file}")
print("Sample qrels entry:", qrels[0:10])
print("Sample query_answers entry:", query_answers[0])