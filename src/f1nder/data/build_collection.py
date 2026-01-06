import json
import os
import argparse


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


def build_collection(input, output):
    inputs = [
        f"{input}/train.json",
        f"{input}/validation.json",
        f"{input}/test.json"
        ]

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=True)
    ap.add_argument("--output_file", default=True)
    args = ap.parse_args()

    build_collection(args.input_dir, args.output_file)