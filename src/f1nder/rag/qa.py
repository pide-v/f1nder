import json
import re
import string
import unicodedata
from argparse import ArgumentParser
from collections import defaultdict
from prompts import rag_system_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM

def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_test(test_queries_path, test_queries_answers_path):
    with open(test_queries_path, "r") as f:
        test = json.load(f)
    with open(test_queries_answers_path, "r") as f:
        answers = json.load(f)
    return test, answers

def load_top_documents(top_documents_path):
    with open(top_documents_path, "r") as f:
        docs = f.readlines()
    return docs

def qa(
        test_queries_path,
        test_queries_answers_path,
        output_path,
        top_documents_path,
        document_collection_path
        ):
    
    #Load test and top documents retrieved by bm25f
    test, answers = load_test(test_queries_path, test_queries_answers_path)
    docs = load_top_documents(top_documents_path)

    #Keep only the top1 document for each test instance
    current = ""
    top_1_docs = []
    for doc in docs:
        splits = doc.split()
        if current != splits[0]:
            current = splits[0]
            top_1_docs.append({"query_id": splits[0], "para_id":splits[2]})
    print(top_1_docs)

    top_5_docs = defaultdict(list)

    for doc in docs:
        splits = doc.split()
        query_id = splits[0]
        para_id = splits[2]

        # Aggiungiamo finché non arriviamo a 5 documenti per query
        if len(top_5_docs[query_id]) < 5:
            top_5_docs[query_id].append(para_id)

    with open(document_collection_path, "r") as f:
        docs_collection = json.load(f)


    #Caricamento modello e tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")

    #Ciclo su tutte le domande del test
    #Poi dovranno arrivare documenti da bm25f

    #10'000 sono davvero tante da runnare in locale, valutiamo se fare un sottoinsieme
    for t in test[:200]:
        question = t['question']
        query_id = t['query_id']

        #search for para_id associated with query_id in top1 retrieved docs
        # Recupero para_id top-5 per la query
        para_ids = top_5_docs.get(query_id, [])

        context = ""
        for i, para_id in enumerate(para_ids):
            doc_text = next(
                doc["context"] for doc in docs_collection if doc["para_id"] == para_id
            )
            context += f"\nDocument {i+1}:\n{doc_text}\n"
        #Preparazione del prompt per RAG

        rag_prompt = rag_system_prompt.format(
            document = normalize_text(context),
            question = normalize_text(question)
        )
        print(rag_prompt)
        messages = [
            {
                "role": "user",
                "content": rag_prompt
            }
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=40)
        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
            )

        #Salvataggio su file della risposta per ciascun query_id
        #Salviamo anche la question in modo da averla a portata di mano
        #quando si fa evaluation con llm as a judge
        obj = {
            "query_id": t['query_id'],
            "answer": answer,
            "question": question
        }
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--test_queries_path")
    ap.add_argument("--test_queries_answers_path")
    ap.add_argument("--output_path")
    ap.add_argument("--top_documents_path")
    ap.add_argument("--document_collection_path")
    args = ap.parse_args()

    qa(
        args.test_queries_path,
        args.test_queries_answers_path,
        args.output_path,
        args.top_documents_path,
        args.document_collection_path
        )