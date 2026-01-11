import json
from argparse import ArgumentParser
import spacy
from spacy.util import is_package
import spacy.cli

# estrae entità da document_collection e crea file document_collection_entities
# che contiene para_id ed entities

# oppure estrae entità dalle queries (ancora da implementare)

ENTITY_LABELS = {"PERSON", "ORG", "GPE", "DATE"}
MODEL = "en_core_web_sm"


def extract_entities_from_text(nlp, text: str):
    if not isinstance(text, str):
        return []

    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            cleaned = ent.text.strip()
            if cleaned:
                entities.append(cleaned)

    # rimuove duplicati ma preserva ordine, ma ha senso farlo? o meglio tenerli?
    return list(dict.fromkeys(entities))


def build_entities(input_file, output_file):
    print(f"Loading documents from {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # forse si puo usare la funzione in io.py

    print(f"Loaded {len(data)} documents")
    print("Loading spaCy model...")
    
    if not is_package(MODEL):
      print(f"Model {MODEL} not found, downloading...")
      spacy.cli.download(MODEL)

    nlp = spacy.load("en_core_web_sm")

    result = []
    for item in data:
        para_id = item.get("para_id")
        text = item.get("context", "")

        if not para_id:
            continue

        ents = extract_entities_from_text(nlp, text)

        result.append({
            "para_id": para_id,
            "entities": ents
        })

    # salva
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(result)} entries to {output_file}")
    print(json.dumps(result[:3], indent=2))


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    build_entities(args.input_file, args.output_file)
