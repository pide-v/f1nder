import json
import argparse


def build_qrels(input_file, qrels_file, answers_file):
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
    print("Sample qrels entry:", qrels[0])
    print("Sample query_answers entry:", query_answers[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", default=True)
    ap.add_argument("--qrels_file", default=True)
    ap.add_argument("--answers_file", default=True)
    args = ap.parse_args()

    build_qrels(args.input_file, args.qrels_file, args.answers_file)