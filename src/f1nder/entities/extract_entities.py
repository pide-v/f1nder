import json
from argparse import ArgumentParser
import spacy
from spacy.util import is_package
import spacy.cli

ENTITY_LABELS = {"PERSON", "ORG", "GPE", "EVENT"}
MODEL = "en_core_web_lg"

def extract_entities_from_text(nlp, text: str):
    if not isinstance(text, str):
        return ""

    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            # normalize entities
            cleaned = "_".join(ent.text.strip().split()).lower()

            if len(cleaned) < 3:
                continue

            if any(c in cleaned for c in ["&", "[", "]", "=", "~"]):
                continue

            if cleaned.isnumeric():
                continue

            entities.append(cleaned)
                
    unique_entities = list(dict.fromkeys(entities))
    return " ".join(unique_entities)


def process_documents(nlp, input_file, output_file):
    print(f"Processing DOCUMENTS from {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        docs = json.load(f)

    result = []

    for item in docs:
        para_id = item.get("para_id")
        text = item.get("context", "")

        if not para_id:
            continue

        ents = extract_entities_from_text(nlp, text)

        result.append({
            "para_id": para_id,
            "context": text,
            "entities": ents
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(result)} document entities to {output_file}")
    print(json.dumps(result[:3], indent=2))


def process_queries(nlp, input_file, output_file):
    print(f"Processing QUERIES from {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    result = []

    for item in queries:
        qid = item.get("query_id")
        question = item.get("question", "")

        if not qid:
            continue

        ents = extract_entities_from_text(nlp, question)

        result.append({
            "query_id": qid,
            "question": question,
            "entities": ents
        })


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(result)} query entities to {output_file}")
    print(json.dumps(result[:3], indent=2))


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--documents_input", required=True)
    ap.add_argument("--documents_output", required=True)
    ap.add_argument("--queries_input", required=True)
    ap.add_argument("--queries_output", required=True)

    args = ap.parse_args()

    if not is_package(MODEL):
        print(f"Model {MODEL} not found, downloading...")
        spacy.cli.download(MODEL)

    print("Loading spaCy model...")
    nlp = spacy.load(MODEL)
    process_documents(nlp, args.documents_input, args.documents_output)
    process_queries(nlp, args.queries_input, args.queries_output)

    print("\n✅ DONE")
