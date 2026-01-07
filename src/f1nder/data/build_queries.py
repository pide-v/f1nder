import json
import re
import unicodedata
import string
from argparse import ArgumentParser


def build_queries(input_file, output_file):
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


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--input_file")
    ap.add_argument("--output_file")
    args = ap.parse_args()

    build_queries(args.input_file, args.output_file)